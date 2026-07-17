<div align="center">

# Slow-Fast Inference (SFI)

**Training-Free Inference Acceleration via Within-Sentence Support Stability**

<a href="https://arxiv.org/abs/2603.12038"><img src="https://img.shields.io/badge/arXiv-2603.12038-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>&nbsp;&nbsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge" alt="License"></a>&nbsp;&nbsp;
<a href="https://github.com/vllm-project/vllm"><img src="https://img.shields.io/badge/vLLM-v1_engine-blueviolet?style=for-the-badge" alt="vLLM"></a>&nbsp;&nbsp;
<a href="https://github.com/vllm-project/flash-attention"><img src="https://img.shields.io/badge/Kernel-FA3%20%2F%20FA4-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="FlashAttention 3 and 4"></a>

<br><br>

<a href="#quick-start">Quick Start</a>&ensp;&middot;&ensp;
<a href="#release-scope">Release Scope</a>&ensp;&middot;&ensp;
<a href="#installation">Installation</a>&ensp;&middot;&ensp;
<a href="#configuration">Configuration</a>&ensp;&middot;&ensp;
<a href="#testing-and-benchmarking">Testing</a>&ensp;&middot;&ensp;
<a href="#troubleshooting">Troubleshooting</a>&ensp;&middot;&ensp;
<a href="#citation">Citation</a>

<br><br>

<img src="assets/motivation_v5.png" width="72%" alt="SFI teaser" />

</div>

> [!IMPORTANT]
> This public CUDA release ships the current sparse runtime and its generated
> FA3/FA4 patch stack. The supported validation path is the one documented in
> this README: prepare the architecture-matched kernel, pass the fresh one-shot
> output/route gate, run official LongBench for quality, and use matched
> sparse/dense pairs for speed.

## Quick Start

The same absolute Python interpreter must own PyTorch, vLLM, the FA3 build or
FA4 JIT, and all helper extensions. Replace the paths below.

```bash
git clone -b cuda-kernel https://github.com/LV-NUS/SFI.git
cd SFI

export PYTHON="/absolute/path/to/environment/bin/python"
export MODEL="/absolute/path/to/qwen3-model"
export GPU="0"
test -x "${PYTHON}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
"${PYTHON}" -c 'import torch, vllm; assert torch.cuda.is_available(); print(torch.__version__)'

# Auto-detect SM80/SM90/SM100 on the selected GPU and prepare FA3 or FA4.
PYTHON="${PYTHON}" bash scripts/setup_flash_attention.sh --arch auto --gpu "${GPU}"

# SM100 only: install CuTe and dependencies from the pinned local metadata.
SFI_CC="$(CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))')"
if [[ "${SFI_CC}" == "10.0" ]]; then
  CUTE_SPEC="${PWD}/third_party_upstreams/vllm-project-flash-attention/flash_attn/cute"
  if "${PYTHON}" -c 'import torch, sys; sys.exit(0 if str(torch.version.cuda).startswith("13.") else 1)'; then
    CUTE_SPEC="${CUTE_SPEC}[cu13]"
  fi
  "${PYTHON}" -m pip install "${CUTE_SPEC}"
fi

# Fresh output/infra/route run: unique artifact tag and fresh helper cache.
RUN_ID="sfi_$(date +%Y%m%d_%H%M%S)"
PYTHON="${PYTHON}" TORCH_EXTENSIONS_DIR="${PWD}/tmp/torch_extensions/${RUN_ID}" MML=16384 bash scripts/run_one_shot.sh "${GPU}" "${MODEL}" "oneshot_${RUN_ID}"

# Self-contained sparse/dense pair on the same exclusive GPU and preset.
WITH_DENSE_REFERENCE=1 PYTHON="${PYTHON}" \
  bash scripts/run_speed.sh "${GPU}" "${MODEL}" bs8x12k sparse "pair_${RUN_ID}"
```

The one-shot run must end with:

```text
child_returncode=0 gate_passed=True production_gate_passed=True producer_gate_passed=True route_proof_passed=True speed_child_route_proof_passed=True decode_tps=...
ONE-SHOT PASS
```

The speed run must end with `SPEED RUN OK`. Its summary contains the adjacent
observer-free dense reference and sparse diagnostic arms; do not splice arms
from different runs, models, shapes, GPU occupancy, or software environments.

### Three independent validation terms

Public validation is intentionally split into three user-facing terms. A
target is accepted only when all applicable terms pass; an HTTP 200 response,
one throughput number, or a dense fallback is not sufficient.

| Term | Entrypoint | Accepted result | What it proves |
|:--|:--|:--|:--|
| One-shot output | `scripts/run_one_shot.sh` | `ONE-SHOT PASS` | Complete output plus producer, lifecycle, backend and real sparse-route evidence on the fixed long workload; it is not the LongBench quality score |
| LongBench quality | `scripts/serve_sparse.sh` then `scripts/run_longbench_v2.sh` | `PASS: official LongBench v2 sparse liveness, completeness and scoring` plus `result.txt` | Official 503-sample score, complete responses, and fresh sparse producer/compact-read evidence in server mode |
| Paired speed | one `WITH_DENSE_REFERENCE=1 scripts/run_speed.sh` run | `SPEED RUN OK` with all three arms accepted | Adjacent same-process-contract sparse speed, observer-free dense reference and sparse diagnostic evidence |

The detailed commands are independent: use [one-shot](#1-fresh-one-shot-output-and-sparse-route-gate),
[paired speed](#2-paired-speed-gate), or [LongBench](#5-longbench-v2-external-evaluation).
One-shot always exercises the fixed long sparse preset. The LongBench runner
automatically sends a long sparse smoke request and rejects the run unless it
observes fresh producer publication and compact-read activity before scoring.

## Overview

SFI accelerates long-context autoregressive decoding by separating attention
into two paths:

| | Fast step | Slow step |
|:--|:--|:--|
| Work | Attend to compact sparse memory | Run dense full attention |
| Timing | Most decode steps | Sentence boundaries or refresh deadlines |
| Purpose | Reuse stable support | Refresh the selected support |

A training-free selector turns dense-attention evidence from slow steps into
compact KV state for later fast steps. The CUDA release adds:

- native FA3 and FA4 mixed-page, dual-source attention paths;
- full-CUDA-graph decode;
- asynchronous refresh, selection, and compact-KV maintenance;
- route, producer, lifecycle, and output gates in the supplied runners.

<div align="center">
<img src="assets/method_new4.png" width="72%" alt="SFI method overview" />
</div>

## Release Scope

| GPU class | Architecture | Kernel | Current evidence | Status |
|:--|:--:|:--|:--|:--|
| NVIDIA A100-class | SM80 | patched FA3 | end-to-end correctness, route proof, determinism and throughput | **Validated** |
| NVIDIA H100/H800-class | SM90 | patched FA3 | patch application, compilation and kernel resource checks | **Build-level support; run the local gates before use** |
| NVIDIA B200-class | SM100 | FA4 CuTe | rebased overlay, sequential patch proof and CuTe fake-JIT contracts | **Packaged; run the target-hardware one-shot gate** |

SM90 and SM100 are not advertised as target-hardware performance-validated
until the one-shot and paired benchmark gates pass on the user's Hopper or
Blackwell machine. The entrypoints still select the correct kernel family
automatically and fail closed on architecture mismatches.

### Patch provenance

The public patch stack contains:

- `kernel_patches/sfi_fa3_sm80_sm90.patch`: shared wrapper plus FA3
  SM80/SM90 implementation;
- `kernel_patches/sfi_fa4_sm100_cute.patch`: a CuTe-only SM100 overlay,
  applied after the FA3 patch.

Both are generated release artifacts and checked against their canonical
source trees. Public users should apply them through
`scripts/setup_flash_attention.sh`; do not hand-edit or reconstruct them from
another checkout.

The setup script:

1. clones the pinned vLLM FlashAttention upstream;
2. applies FA3 for SM80/SM90, or FA3 followed by FA4 for SM100;
3. initializes CUTLASS;
4. builds `vllm_flash_attn/_vllm_fa3_C*.so` for SM80/SM90, while SM100 keeps
   the patched CuTe sources for runtime JIT.

SFI does not modify the installed vLLM or flash-attention packages. The
patched clone is loaded at process startup through
`VLLM_SPARSE_FA3_UPSTREAM_ROOT`.

## Installation

### Prerequisites

| Dependency | Requirement |
|:--|:--|
| Python | 3.10 or newer; use one executable absolute path throughout |
| PyTorch | CUDA build compatible with the installed driver/toolkit |
| vLLM | `0.19.x` v1 engine; SFI fails closed on incompatible private-API drift |
| CUDA toolkit | toolkit supported by the chosen PyTorch/vLLM build and target architecture, with `nvcc` available |
| Build tools | `git`, `cmake`, `ninja`, and a supported host compiler |
| Model | current automated release gates target Qwen3-family checkpoints |

PyTorch, the CUDA toolkit, the NVIDIA driver, and the target GPU architecture
must be mutually compatible. Because SFI patches vLLM v1 private interfaces,
the supplied preflight and one-shot gates are authoritative: any unsupported
API or binary-ABI drift must fail before a benchmark or server is accepted.

Create an isolated user environment if one is not already available. The
following installs the supported vLLM line and build helpers; choose a
CUDA-enabled vLLM/PyTorch wheel compatible with the target system when the
default package index is not appropriate:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "vllm>=0.19,<0.20" ninja cmake
export PYTHON="$(python -c 'import os, sys; print(os.path.realpath(sys.executable))')"
```

SM100 additionally requires a driver/toolkit stack that supports `sm_100a`.
After setup creates the patched checkout, install its CuTe package from the
pinned local metadata as shown in the next section. That metadata currently
requires `nvidia-cutlass-dsl>=4.4.2` together with the matching CuTe runtime
dependencies. The one-shot and server preflights reject a target that cannot
provide that path.

Before building, verify the environment:

```bash
export PYTHON="/absolute/path/to/environment/bin/python"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export CUDA_HOME="$(realpath -e /absolute/path/to/cuda-12.x)"
export CUDA_PATH="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:$(dirname "${PYTHON}"):${PATH}"

"${PYTHON}" - <<'PY'
import shutil
import torch
import vllm

assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
assert shutil.which("nvcc"), "nvcc is not on PATH"
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("vllm:", vllm.__version__)
PY
```

### Prepare the architecture-matched kernel

Use the selected GPU as the source of truth:

```bash
export PYTHON="/absolute/path/to/environment/bin/python"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export CUDA_HOME="$(realpath -e /absolute/path/to/cuda-12.x)"
export CUDA_PATH="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:$(dirname "${PYTHON}"):${PATH}"
export GPU="0"

PYTHON="${PYTHON}" NVCC_THREADS=4 MAX_JOBS=8 \
  bash scripts/setup_flash_attention.sh --arch auto --gpu "${GPU}"
```

`--arch auto` maps compute capability 8.0 to SM80/FA3, 9.0 to SM90/FA3,
and 10.0 to SM100/FA4. An explicit `--arch sm80`, `sm90`, or `sm100` is
available for controlled automation; when `--gpu` is also supplied, a
mismatch fails closed.

SM80/SM90 setup applies the FA3 patch and builds
`vllm_flash_attn/_vllm_fa3_C*.so`. SM100 setup applies the shared FA3 patch
and then the FA4 CuTe overlay; it prepares runtime-JIT sources and therefore
does not require an FA3 shared object. Every path emits
`sfi_flash_attention_build_provenance.json` in the patched checkout.
That provenance owns the canonical `CUDA_HOME`, `CUDA_PATH`, `CUDACXX`,
compiler release and, for FA3, the actual CMake compiler/root. One-shot,
speed, and server entrypoints validate it before deriving helper-cache
identities or starting JIT; conflicting caller CUDA variables fail closed.

To inspect the patch application manually, reproduce the same source order
against the pinned upstream base printed by the setup script:

```bash
export SFI_ROOT="${PWD}"
export FA_ROOT="${SFI_ROOT}/third_party_upstreams/manual-flash-attention"
export BASE_COMMIT="f5bc33cfc02c744d24a2e9d50e6db656de40611c"

git clone https://github.com/vllm-project/flash-attention.git "${FA_ROOT}"
git -C "${FA_ROOT}" checkout --detach "${BASE_COMMIT}"
git -C "${FA_ROOT}" apply --check "${SFI_ROOT}/kernel_patches/sfi_fa3_sm80_sm90.patch"
git -C "${FA_ROOT}" apply --index "${SFI_ROOT}/kernel_patches/sfi_fa3_sm80_sm90.patch"
git -C "${FA_ROOT}" submodule update --init csrc/cutlass
```

For an SM100 source-replay audit, apply the CuTe overlay after FA3:

```bash
git -C "${FA_ROOT}" apply --check "${SFI_ROOT}/kernel_patches/sfi_fa4_sm100_cute.patch"
git -C "${FA_ROOT}" apply --index "${SFI_ROOT}/kernel_patches/sfi_fa4_sm100_cute.patch"
```

These manual commands are inspection-only: they do not emit the build
provenance required by the supplied one-shot, speed, server, or LongBench
runners. For every executable validation or deployment, use
`scripts/setup_flash_attention.sh` and point the runner at that setup-generated
target. The runner rejects a hand-built or stale target before launching a
workload.

The first one-shot run also compiles small selector/bounds CUDA extensions.
Their cache must be writable and must never be shared across incompatible
Python, PyTorch, CUDA, or GPU-architecture environments. The supplied runners
derive ABI- and semantic-version-partitioned caches automatically.

### Runtime injection

Python imports `sitecustomize.py` when the repository root is on
`PYTHONPATH`. The installation chain is:

```text
sitecustomize.py
  -> patches/fa3_native/install.py
     -> load the architecture-matched patched clone and bridge it into vLLM
  -> patches/patch_installer.py
     -> install the sparse controller and v1 runner hooks
  -> VLLMSparseController
     -> select dense-refresh or compact sparse attention for each step
```

The supplied offline runners configure this chain. For manual integration,
the critical identities are:

| Variable | Required value or meaning |
|:--|:--|
| `PYTHON` | absolute executable used for every parent and child process |
| `PYTHONPATH` | contains the SFI repository root |
| `VLLM_SPARSE_FA3_UPSTREAM_ROOT` | patched FlashAttention clone; the compatibility name is used for both FA3 and FA4 |
| `VLLM_ATTENTION_BACKEND` | exactly `FLASH_ATTN_VLLM_V1` |
| `VLLM_FLASH_ATTN_VERSION` | `3` for SM80/SM90; `4` for SM100 |
| `SFI_CUDA_ARCH` | detected or explicit `sm80`, `sm90`, or `sm100` |
| `SFI_ATTENTION_KERNEL` | `fa3-native` for SM80/SM90; `fa4-cute` for SM100 |
| `VLLM_SPARSE_CONTROLLER_JSON` | enables and configures the controller |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` |
| `TORCH_EXTENSIONS_DIR` | ABI-partitioned writable helper-extension cache |
| `VLLM_KV_CACHE_MEMORY_BYTES` | explicit per-GPU KV pool size when benchmarking |

If `VLLM_SPARSE_CONTROLLER_JSON` is absent, the sparse controller is not
installed. If the patched FlashAttention root is also absent, execution is
normal vLLM.

## Configuration

The runners generate controller JSON automatically. This representative
quality configuration documents the public contract:

```json
{
  "enabled": true,
  "attn_mode": "compact_recent",
  "compact_page_residency_enabled": true,
  "max_live_sparse_slots": 8,
  "compact_blocks_per_slot": 259,
  "k_min": 32,
  "k_max": null,
  "sink": 4,
  "recent": 256,
  "refresh_interval": 96,
  "refresh_coalesce_window": 0,
  "alpha_fair": {"k_head": 4096},
  "prefill_last_n_query": 2,
  "one_shot_bootstrap_only": true,
  "continuous_producer_enabled": true,
  "trigger": {
    "refresh_interval": 96,
    "enable_sentence_triggers": true,
    "min_refresh_gap": 24,
    "sentence_cooldown": 2
  }
}
```

| Field | Meaning |
|:--|:--|
| `alpha_fair.k_head` | selected tokens per KV head; 4096 favors quality, while 1536–2048 favors speed |
| `max_live_sparse_slots` | maximum simultaneous sparse requests; match real concurrency |
| `compact_blocks_per_slot` | 16-token compact pages reserved per slot |
| `sink` / `recent` | always-retained prefix and trailing tokens |
| `refresh_interval` | maximum decode steps between dense refreshes |
| `trigger.enable_sentence_triggers` | additionally refresh at sentence boundaries |
| `prefill_last_n_query` | trailing prompt rows used for bootstrap selection |
| `one_shot_bootstrap_only` and `continuous_producer_enabled` | bootstrap once, then maintain sparse state asynchronously |

The current compact runtime aligns retained tokens to a 112-token tile. The
allocation must satisfy:

```text
required_blocks = ceil(align_up(sink + k_head, 112) / 16)
compact_blocks_per_slot >= required_blocks
```

For `sink=4` and `k_head=4096`, the minimum is 259 blocks. Configuration
validation rejects undersized or unknown fields.

### GPU memory budgeting

SFI state, CUDA-graph capture buffers, and selector workspaces live outside
vLLM's KV-pool budget. Plan memory as:

```text
model weights + VLLM_KV_CACHE_MEMORY_BYTES + SFI/JIT/capture headroom < VRAM
```

The sparse KV pool must hold both full-history KV and the compact-page lease.
With dual-generation compact state enabled:

```text
compact_lease_bytes =
    slots * blocks_per_slot * 16 * kv_bytes_per_token * 2

capacity_tokens_per_request =
    (KVB - compact_lease_bytes) / (kv_bytes_per_token * batch)

capacity_tokens_per_request > context_tokens + max_new_tokens
```

`KVB` is per GPU. For tensor parallelism, use the per-rank
`kv_bytes_per_token` after KV-head sharding. An undersized pool can make the
scheduler serialize requests and recompute prefill, roughly doubling decode
steps without an immediate OOM. `run_speed.sh` checks this before launch.

The speed runner derives `kv_bytes_per_token` from `MODEL/config.json` for
every TP tier:

```text
KV_TOKEN_BYTES = layers * KV_heads_per_rank * head_dim * 2(K+V) * dtype_bytes
```

If `KV_TOKEN_BYTES` is supplied for audit compatibility, it must exactly equal
the derived value. KV heads that cannot shard evenly across `TP` fail closed;
the runner never guesses or accepts a memory value for a different model.

Use these controls in order:

1. set `MML` to the real prompt-plus-generation requirement;
2. pin `KVB` rather than relying on utilization-based auto-sizing;
3. set slots to actual concurrency;
4. choose the smallest valid blocks-per-slot for the selected `k_head`;
5. leave additional headroom for JIT compilation and CUDA graph capture.

## Testing and Benchmarking

### 1. Fresh one-shot output and sparse-route gate

```bash
export PYTHON="/absolute/path/to/environment/bin/python"
export MODEL="/absolute/path/to/qwen3-model"
RUN_ID="oneshot_$(date +%Y%m%d_%H%M%S)"

PYTHON="${PYTHON}" TORCH_EXTENSIONS_DIR="${PWD}/tmp/torch_extensions/${RUN_ID}" MML=16384 bash scripts/run_one_shot.sh 0 "${MODEL}" "oneshot_${RUN_ID}"
```

This infrastructure gate verifies fresh child completion, mixed-page kernel routing,
producer activity, sparse lifecycle invariants, output health, and summary
freshness. SM90 always enables its dense reference because the Hopper
correctness gate is reference-backed. On SM80/SM100, optional dense output
inspection is available with a fourth `--with-reference` argument:

```bash
PYTHON="${PYTHON}" MML=16384 bash scripts/run_one_shot.sh 0 "${MODEL}" "oneshot_reference_${RUN_ID}" --with-reference
```

The `bs2long-cap128` preset intentionally uses the tracked
`benchmarks/needle_prompt_two_parts.txt` calibration fixture. Its two requests
preserve the previously validated 11,262/7,456-token workload and 128-token
decode cap. Do not replace the prompt, output length, or sparse trigger
parameters when comparing a change with the established one-shot result.

The textual comparator is intentionally strict; a semantic text difference is
informational, while route, producer, lifecycle, and process failures remain
hard failures. Treat the official LongBench score, not one-shot textual parity,
as the public quality result.

### 2. Paired speed gate

```bash
export PYTHON="/absolute/path/to/environment/bin/python"
export MODEL="/absolute/path/to/qwen3-model"
PAIR_ID="pair_$(date +%Y%m%d_%H%M%S)"

WITH_DENSE_REFERENCE=1 PYTHON="${PYTHON}" \
  bash scripts/run_speed.sh 0 "${MODEL}" bs8x12k sparse "${PAIR_ID}"
```

The built-in tiers are A100-40GB starting points:

| Tier | Batch × context | KV pool | Max model length |
|:--|:--:|:--:|:--:|
| `bs8x12k` | 8 × 12k | 18 GiB | 16384 |
| `bs8x16k` | 8 × 16k | 22 GiB | 20480 |
| `bs4x24k` | 4 × 24k | 16 GiB | 28672 |
| `bs2x30k` | 2 × 30k | 16 GiB | 36864 |

Each run creates or reuses a tokenizer-bound, content-addressed corpus with
exactly `BS × CTX` tokens derived from the tracked fixed calibration source.
The source hash, tokenizer fingerprint, and corpus hash are part of the
artifact identity. Dense/sparse comparisons must use the same generated corpus
and all other workload parameters; changing the corpus starts a new baseline.

On SM90/SM100, another model, or a GPU with different memory capacity, treat
the tier only as a workload shape and override `BS`, `CTX`, `KVB`, `MML`, and
`MAX_NEW` for the target. The runner derives per-rank KV bytes from the model,
detects the architecture, and selects FA3 or FA4; it does not infer a safe
memory budget for an unfamiliar model or GPU.

Only compare runs that use:

- the same model, exact corpus, batch, context, generation length, and KV pool;
- the same code, architecture-matched kernel identity, interpreter, selector
  cache identity, and GPU set;
- an exclusive idle GPU with no overlapping process;
- complete outputs, expected decode length, route proof, and zero unexpected
  fallback.

Use at least three independent self-contained pairs for a performance claim.
Each pair has the fixed order `sparse_speed`, `dense_reference`, then
`sparse_diagnostic`; never combine standalone arms. `decode_tps` is the
end-to-end decode-window metric.
`all_decode_tps` isolates the window after every request has completed
chunked prefill; report both rather than selecting the more favorable one.

### 3. Tensor parallelism

The speed runner accepts `TP=N` and a comma-separated GPU list. The number
of listed GPUs must equal `TP`, and `KVB` remains per GPU:

```bash
export PYTHON="/absolute/path/to/environment/bin/python"
export MODEL="/absolute/path/to/qwen3-model"

WITH_DENSE_REFERENCE=1 PYTHON="${PYTHON}" TP=2 BS=2 CTX=128000 \
  KVB=20401094656 MML=132096 \
  bash scripts/run_speed.sh "0,1" "${MODEL}" bs2x30k sparse "tp2_pair"
```

The runner probes topology and avoids vLLM custom all-reduce on unsupported
PCIe-only layouts. NVLink availability, collective overhead, and per-rank KV
capacity can dominate TP results; treat TP as a separate paired validation,
not as evidence inherited from a single-GPU run.

The exact SM80 TP8 remote gate is a single self-contained sparse invocation.
It runs the adjacent observer-free dense reference internally, requires an
exact clean release identity and model configuration, and rejects incomplete
rank-local route, lifecycle, output, or custom-all-reduce evidence:

```bash
export SFI_EXPECTED_GIT_COMMIT="$(git rev-parse HEAD)"
export SFI_EXPECTED_MODEL_CONFIG_SHA256="$(sha256sum "${MODEL}/config.json" | awk '{print $1}')"
export VLLM_SPARSE_FA3_UPSTREAM_ROOT="${PWD}/third_party_upstreams/vllm-project-flash-attention"
export TORCH_EXTENSIONS_DIR="${PWD}/tmp/torch_extensions/tp8_${SFI_EXPECTED_GIT_COMMIT}"
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF

PYTHON="${PYTHON}" TP=8 bash scripts/run_speed.sh \
  "0,1,2,3,4,5,6,7" "${MODEL}" tp8x64k sparse \
  "tp8_exact_${SFI_EXPECTED_GIT_COMMIT:0:12}"
```

The run is accepted only when `scripts/check_run_speed_summary.py` prints
`SPEED RUN OK`. This is the remote target-hardware gate; the public release
does not claim TP8 acceleration before that exact run passes.

### 4. Sparse serving

After setup and the one-shot gate, start the fail-closed OpenAI-compatible
server in a dedicated shell:

```bash
export PYTHON="/absolute/path/to/environment/bin/python"
export MODEL="/absolute/path/to/qwen3-model"
export GPU="0"

PYTHON="${PYTHON}" HOST=127.0.0.1 MML=32768 SLOTS=8 \
  bash scripts/serve_sparse.sh "${GPU}" "${MODEL}" 8000
```

The launcher enumerates every selected rank, requires one homogeneous exact
SM80/SM90/SM100 capability, binds the corresponding FA3/FA4 kernel, makes
`--max-num-seqs` equal to sparse slots, includes that size in CUDA-graph
capture, partitions helper caches by ABI and selector semantics, and writes a
PID-bound manifest plus fresh route/liveness artifacts under
`tmp/serve_runs/`. It refuses an occupied address rather than killing an
unrelated process.

The default bind is loopback-only and may use the documented local test key.
To expose the server on another interface, set an explicit non-loopback
`HOST` and a strong `API_KEY`; the launcher rejects the default key outside
loopback:

```bash
PYTHON="${PYTHON}" HOST=0.0.0.0 API_KEY="replace-with-a-strong-secret" \
  bash scripts/serve_sparse.sh "${GPU}" "${MODEL}" 8000
```

For an API reachability check, call the endpoint from another shell:

```bash
export MODEL="/absolute/path/to/qwen3-model"

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer token-abc123' \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Summarize the supplied context.\"}],\"temperature\":0,\"max_tokens\":128}"
```

The request model must match an ID returned by `/v1/models`; when a local
path is served, that ID is normally the same path. This deliberately short
request checks only API reachability and is not expected to cross the sparse
threshold. The LongBench command below automatically sends the long smoke and
checks fresh manifest-bound sparse activity before the official evaluation;
users do not need to tune a prompt or lower a trigger threshold to prove the
server path.

### 5. LongBench v2 external evaluation

SFI does not vendor the complete LongBench evaluation dataset or harness.
The tracked calibration fixtures used by one-shot/speed are regression inputs,
not an embedded LongBench evaluation. `scripts/run_longbench_v2.sh` binds the
official [THUDM/LongBench](https://github.com/THUDM/LongBench) prediction and
scoring workflow to a live SFI sparse server and its route evidence.

First prepare a separate official checkout and a lightweight client
environment. Keeping the client separate prevents its dependency choices from
changing the SFI server environment:

```bash
export PYTHON="/absolute/path/to/sfi-server-environment/bin/python"
export LONGBENCH_ROOT="/absolute/path/to/LongBench"
git clone https://github.com/THUDM/LongBench.git "${LONGBENCH_ROOT}"

"${PYTHON}" -m venv --system-site-packages \
  "${LONGBENCH_ROOT}/.venv-sfi-client"
export LONGBENCH_PYTHON="${LONGBENCH_ROOT}/.venv-sfi-client/bin/python"
"${LONGBENCH_PYTHON}" -m pip install --upgrade pip
"${LONGBENCH_PYTHON}" -m pip install datasets openai transformers tiktoken tqdm
git -C "${LONGBENCH_ROOT}" rev-parse HEAD
```

Do not install the external checkout's requirements into the SFI server
environment. The client only needs the packages imported by the official
prediction script; the server continues to use the supported vLLM stack.

Add one model alias to the official `config/model2path.json` and
`config/model2maxlen.json`. The mapped path must be the same absolute model
path served by SFI. This example uses a 120,000-token LongBench input budget
under a 131,072-token server limit:

```bash
export MODEL="/absolute/path/to/qwen3-model"
export MODEL_NAME="Qwen3-local"
export LONGBENCH_MAXLEN="120000"

"${LONGBENCH_PYTHON}" - \
  "${LONGBENCH_ROOT}" "${MODEL_NAME}" "${MODEL}" "${LONGBENCH_MAXLEN}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
name = sys.argv[2]
model = str(pathlib.Path(sys.argv[3]).resolve(strict=True))
maxlen = int(sys.argv[4])
for filename, value in (
    ("model2path.json", model),
    ("model2maxlen.json", maxlen),
):
    path = root / "config" / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    data[name] = value
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
```

The official `pred.py` defaults currently target
`http://127.0.0.1:8000/v1` with the local key `token-abc123`, which matches the
default SFI launcher below. If you intentionally change the host, port, or
key, update `URL` and `API_KEY` in the external `pred.py` as described by the
official LongBench README; the SFI experiment shell checks the values before
running and does not edit the external checkout.

Stop any earlier short-context server, then start SFI with sufficient model
length and KV capacity. Reduce `SLOTS` if the target cannot hold this context,
but keep it at least as large as the requested LongBench concurrency:

```bash
export PYTHON="/absolute/path/to/sfi-server-environment/bin/python"
export MODEL="/absolute/path/to/qwen3-model"

PYTHON="${PYTHON}" HOST=127.0.0.1 MML=131072 SLOTS=1 \
  bash scripts/serve_sparse.sh "0" "${MODEL}" 8000
```

Run the external harness from a second shell:

```bash
export PYTHON="/absolute/path/to/sfi-server-environment/bin/python"
export LONGBENCH_ROOT="/absolute/path/to/LongBench"
export LONGBENCH_PYTHON="${LONGBENCH_ROOT}/.venv-sfi-client/bin/python"
export MODEL_NAME="Qwen3-local"

PYTHON="${PYTHON}" LONGBENCH_ROOT="${LONGBENCH_ROOT}" \
  LONGBENCH_PYTHON="${LONGBENCH_PYTHON}" \
  bash scripts/run_longbench_v2.sh "${MODEL_NAME}" 1 8000
```

The shell verifies the live server PID, command, controller, model/MML,
authentication, all TP ranks, architecture, and FA3/FA4 family. It performs a
long sparse smoke and a fresh producer/compact-read liveness check, invokes
the external official `pred.py`, checks a second liveness delta, requires 503
unique non-empty responses, and runs the external official `result.py` in an
isolated result directory. Artifacts include the external Git revision,
configuration hashes, server identity, predictions, liveness logs, and
official score output under `tmp/longbench_v2_runs/`. The final lines print
the exact `score=.../result.txt` and `score_summary=.../score_summary.json`
paths. SFI independently recomputes Overall/Easy/Hard/Short/Medium/Long from
the 503 prediction rows and rejects a malformed or mismatched official score.

Start with `SLOTS=1` and `N_PROC=1`; increase both only after the KV budget and
target-hardware gate prove that the longer concurrent workload fits.
`N_PROC` must not exceed server `SLOTS`. The configured LongBench maximum plus
generation and chat-template overhead must fit `MML`; changing the truncation
length, prompt, sampling, model, upstream revision, dataset state, or scorer
changes the experiment and must be reported. The upstream client does not pin
a dataset revision, so the recorded Git/source hashes alone are not a dataset
snapshot. For a dense comparison, use the same official checkout, config,
model, sampling, and dataset state against a dense vLLM server, store it in a
separate result directory, and follow the official scoring instructions. Never
accept a sparse quality result when either liveness gate or output completeness
fails.

## Repository Layout

```text
SFI/
├── kernel_patches/
│   ├── sfi_fa3_sm80_sm90.patch    # generated shared/FA3 patch
│   └── sfi_fa4_sm100_cute.patch   # generated SM100 CuTe overlay
├── patches/                        # runtime controller and FA3/FA4 bridge
├── scripts/
│   ├── setup_flash_attention.sh    # detect, clone, apply, build/JIT-ready
│   ├── run_one_shot.sh             # correctness and route gate
│   ├── run_speed.sh                # paired performance runner
│   ├── check_tp8_arm_teardown.py    # post-arm worker/GPU lifecycle gate
│   ├── serve_sparse.sh              # fail-closed sparse API server
│   ├── run_longbench_v2.sh          # external LongBench + sparse-route gate
│   └── check_sparse_liveness.py    # server sparse-activity judge
├── benchmarks/                     # offline end-to-end and kernel runners
├── utils/                          # selector/bounds CUDA extensions
├── hybrid_selectors/               # training-free selector
├── triton_kernel/                  # selector-side helpers, not attention backend
├── sitecustomize.py                # process-start injection
└── assets/
```

## Troubleshooting

| Symptom | Cause and action |
|:--|:--|
| `PYTHON env required` or wrong extension ABI | export one executable absolute `PYTHON`; do not mix environments or reuse a cache built by another ABI |
| setup fails before compilation or JIT preparation | verify the selected GPU, `CUDA_HOME`, `nvcc --version`, host compiler, `ninja`, PyTorch CUDA visibility, and writable build paths |
| no `_vllm_fa3_C*.so` after SM80/SM90 setup | the FA3 build did not complete; rerun setup and do not launch until the shared object exists |
| `ModuleNotFoundError` under `patches.*`, `utils.model_kv_contract`, or `scripts.*` | the release checkout is incomplete; use a clean current `cuda-kernel` commit and never create local stubs |
| SM100 setup has no `_vllm_fa3_C*.so` | expected: SM100 uses the patched FA4 CuTe runtime-JIT source; require the one-shot/FA4 preflight instead |
| helper-extension JIT fails | ensure `ninja` exists and `TORCH_EXTENSIONS_DIR` is writable and ABI-isolated |
| architecture mismatch or heterogeneous TP | every selected rank must have exact homogeneous CC 8.0, 9.0, or 10.0; make explicit `SFI_CUDA_ARCH` match it |
| no mixed-page route proof | verify repository root in `PYTHONPATH`, the patched FlashAttention root, backend `FLASH_ATTN_VLLM_V1`, FA version 3/4 for the target, and valid controller JSON |
| server rejects non-loopback bind | export a non-default `API_KEY`; the documented test key is accepted only on loopback |
| LongBench checkout/client error | clone the official THUDM repository, use an absolute `LONGBENCH_ROOT`, and install client packages in `LONGBENCH_PYTHON` rather than changing the SFI server environment |
| LongBench URL, key, model, or max-length mismatch | make external `pred.py` and both official config maps match the live server; keep configured input plus generation/template headroom within `MML` |
| engine-start OOM | lower `MML`, explicitly reduce `KVB`, reduce batch/slots, and reserve JIT/CUDA-graph headroom |
| decode steps are about 2× expected | KV pool cannot hold full KV plus compact dual-generation lease; raise `KVB` or reduce context, generation length, batch, or blocks |
| throughput is unstable or unexpectedly low | reserve an exclusive GPU, check clocks/power, repeat alternating pairs, and reject runs with incomplete outputs or fallback |
| TP worker fails in custom all-reduce | verify topology and do not force custom all-reduce on an unsupported PCIe-only configuration |
| stale artifacts appear to pass | use unique tags and timestamped artifact paths; require summaries and liveness files newer than run start |

## Results

The paper reports 1.6–14.4× end-to-end decode acceleration across its
long-context workloads while preserving quality close to dense full
attention. Those paper numbers are algorithm-level results; they are not a
substitute for the paired release benchmark on the user's hardware.

<div align="center">
<img src="assets/speed.png" width="72%" alt="SFI throughput results" />
</div>

For the public CUDA release, use the emitted one-shot and paired-speed
artifacts as the deployment verdict. Performance depends on model shape,
context length, batch size, sparse retention, memory capacity, and GPU
topology.

## Roadmap

- [x] FA3 native sparse attention path for SM80
- [x] SM80 end-to-end correctness and throughput gates
- [x] SM90 patch application, compilation, and resource checks
- [x] FA4/SM100 overlay rebase and sequential FA3→FA4 patch proof
- [x] architecture-adaptive setup, one-shot, sparse serving, and external LongBench gate
- [ ] SM90 target-hardware end-to-end validation
- [ ] SM100 target-hardware end-to-end and paired performance validation
- [ ] TP8 server/LongBench target-hardware validation
- [ ] SGLang backend support

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

For questions, open an [issue](https://github.com/LV-NUS/SFI/issues) or email
[xyxie@pku.edu.cn](mailto:xyxie@pku.edu.cn).
