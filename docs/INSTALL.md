# Installation Guide — CUDA Kernel Edition

SFI integrates into vLLM via **runtime patching** (no vLLM source
modification) and runs its fast path on **natively patched FlashAttention
kernels** (FA3 for SM80/SM90, FA4 CuTe for SM100). Everything below is
copy-paste runnable; each step states its success criterion.

<br>

## Prerequisites

| Dependency | Requirement |
|:-----------|:------------|
| Python | ≥ 3.10 |
| PyTorch | CUDA build matching your driver (`torch.cuda.is_available()` is `True`) |
| vLLM | a **v1-engine** release; SFI patches vLLM's v1 engine internals (attention entry point + model-runner step hooks) at runtime, so significant vLLM API drift may need adaptation |
| CUDA toolkit | ≥ 12.0 with `nvcc` on `PATH` (set `CUDA_HOME`); Blackwell/FA4 needs ≥ 12.8 |
| Build tools | `cmake`, `ninja` (FA3 kernel build) |
| Misc | `transformers` (tokenizer for corpus generation), `git` |

> [!NOTE]
> Exact Python/PyTorch/CUDA versions differ per machine and are intentionally
> not pinned here. The rule is: your PyTorch, your CUDA toolkit, and your GPU
> driver must agree with each other; SFI adapts to the rest.

<br>

## Step 1 — Clone

```bash
git clone -b cuda-kernel https://github.com/LV-NUS/SFI.git
cd SFI
```

<br>

## Step 2 — Build the patched FlashAttention kernels

One script clones the pinned upstream, applies the SFI kernel patches, and
builds:

```bash
# A100 / H100 (FA3):
bash scripts/setup_flash_attention.sh

# B200 (adds the FA4 CuTe patch; FA4 kernels JIT-compile at runtime):
bash scripts/setup_flash_attention.sh --with-fa4
```

What to expect:

- The clone lands in `third_party_upstreams/vllm-project-flash-attention/`
  (SFI's default search location — keep it there unless you have a reason).
- `TORCH_CUDA_ARCH_LIST` is auto-detected from GPU 0
  (`8.0` on Ampere, `9.0a` on Hopper). Override it to cross-compile.
- The FA3 build is the slow step: **~25–40 minutes** for a full build.
  `NVCC_THREADS`/`MAX_JOBS` control parallelism.
- **Success criterion**: the script ends with
  `FA3 OK: .../vllm_flash_attn/_vllm_fa3_C.abi3.so`.

Details of the patch set: [`kernel_patches/README.md`](../kernel_patches/README.md).

<br>

## Step 3 — Understand the injection chain (nothing to install)

```
Python process start
      │
      ▼
sitecustomize.py            PYTHONPATH includes the SFI root → auto-imported
      │                     in every process, including vLLM workers
      ▼
patches/fa3_native/install.py
      │                     if VLLM_SPARSE_FA3_UPSTREAM_ROOT is set: loads the
      │                     patched clone and bridges its kernels into vLLM's
      ▼                     flash-attn interface (prebuilt _vllm_fa3_C*.so —
                            FA3, or CuTe JIT — FA4); dense runs use it too
patches/patch_installer.py  if VLLM_SPARSE_CONTROLLER_JSON is set:
      │                     monkey-patches vLLM's attention entry point and
      │                     the v1 model-runner step hooks (_prepare_inputs /
      ▼                     _dummy_run / _update_states); creates the controller
VLLMSparseController        routes every decode step: fast (sparse) or
                            slow (dense refresh)
```

The provided scripts set `PYTHONPATH` and all env vars themselves. For manual
runs, the minimum is:

```bash
export PYTHONPATH="/path/to/SFI:${PYTHONPATH}"
export VLLM_SPARSE_FA3_UPSTREAM_ROOT="/path/to/SFI/third_party_upstreams/vllm-project-flash-attention"
export VLLM_SPARSE_CONTROLLER_JSON='{...}'        # see Configuration below
```

A handful of small selector/bounds CUDA helpers JIT-compile on first run into
`TORCH_EXTENSIONS_DIR` (a couple of minutes, cached afterwards, arch
auto-detected from the live GPU).

<br>

## Step 4 — Verify

```bash
bash scripts/run_one_shot.sh 0 /path/to/any-qwen3-model
```

This runs the full sparse pipeline (CUDA-graph decode + async refresh +
selector) on a built-in 2-sequence long-context preset, then checks the
production gate: kernel-route proof, producer contract, output health, and
lifecycle invariants.

**Success criterion** — the script prints:

```
gate_passed=True producer_gate_passed=True decode_tps=<number>
ONE-SHOT PASS
```

and exits 0. Add `--with-reference` (as the 4th argument) to also run a dense
pass and print an answer-level sparse-vs-dense comparison — informational, not
gated: the strict comparator rarely matches chain-of-thought outputs verbatim
even when the final answers agree. For models whose native context exceeds
your card's KV budget, set `MML` (e.g. `MML=16384` for a 4B model on 40 GB).

<br>

---

<br>

## Configuration

SFI is configured by one JSON env var, `VLLM_SPARSE_CONTROLLER_JSON`:

```json
{
  "enabled": true,
  "attn_mode": "compact_recent",
  "compact_page_residency_enabled": true,
  "max_live_sparse_slots": 8,
  "compact_blocks_per_slot": 288,
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
    "min_refresh_gap": 16,
    "sentence_cooldown": 2
  }
}
```

| Field | Meaning |
|:------|:--------|
| `alpha_fair.k_head` | tokens selected per KV head — the main quality/speed dial. **4096** for evaluation quality, **1536–2048** for throughput |
| `max_live_sparse_slots` | concurrent sparse requests; size it to your batch |
| `compact_blocks_per_slot` | compact-KV pages (16 tokens each) per request; must hold `k_head` (`>= k_head/16`) |
| `sink` / `recent` | always-kept head/tail tokens of the sequence |
| `refresh_interval` | dense-refresh cadence in decode steps |
| `trigger.enable_sentence_triggers` | also refresh at sentence boundaries (natural semantic shift points) |
| `prefill_last_n_query` | trailing prompt rows used to bootstrap the first selection |
| `one_shot_bootstrap_only` + `continuous_producer_enabled` | production mode: one-shot bootstrap, then continuous async refresh |

Unknown fields are rejected (fail-closed), so start from this template.
`scripts/serve_sparse.sh` generates it for you from env knobs.

### Runtime environment variables

| Variable | Purpose |
|:---------|:--------|
| `VLLM_SPARSE_CONTROLLER_JSON` | enables + configures SFI (absence = vanilla vLLM) |
| `VLLM_SPARSE_FA3_UPSTREAM_ROOT` | path to the patched flash-attention clone |
| `VLLM_ATTENTION_BACKEND` | `FLASH_ATTN` (serve) / `FLASH_ATTN_VLLM_V1` (offline runners set it themselves) |
| `VLLM_FLASH_ATTN_VERSION` | `3` on SM80/SM90, `4` on SM100 |
| `VLLM_SPARSE_ASYNC_REFRESH=1` | async refresh workers (production) |
| `VLLM_SPARSE_ONE_SHOT_ASYNC_BOOTSTRAP=1` | async one-shot bootstrap (production) |
| `VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_BATCHED_FLUSH=1` | batched refresh flush under full-CUDA-graph replay |
| `VLLM_SPARSE_FULL_CUDAGRAPH_REPLAY_REFRESH_DEFER_TO_DEADLINE=1` | defer refresh work to the step deadline |
| `VLLM_SPARSE_REFRESH_ENQUEUE_STAGGER=1` | stagger refresh enqueue across requests |
| `VLLM_KV_CACHE_MEMORY_BYTES` | pin vLLM's KV pool size (see memory budgeting) |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | required with CUDA multiprocessing |
| `VLLM_SPARSE_SITE_LOG=1` | write patch-install evidence to `/tmp/vllm_sparse_site.log` |
| `VLLM_SPARSE_FA3_ROUTE_TRACE_LOG=<file>` | per-step kernel-routing trace (debugging/proof) |
| `TORCH_EXTENSIONS_DIR` | cache dir for the JIT helper extensions |

vLLM server flags that must accompany SFI:
`--compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[...]}'`
and `--disable-cascade-attn` (the offline benchmark runners pass their
equivalents automatically).

<br>

## Adapting to your machine

**GPU architecture** decides the kernel family and two env values:

| Your GPU | `TORCH_CUDA_ARCH_LIST` (build) | `VLLM_FLASH_ATTN_VERSION` (run) |
|:--|:--|:--|
| A100 / SM80-class Ampere | `8.0` | `3` |
| H100 / H800 / SM90 Hopper | `9.0a` | `3` |
| B200 / SM100 Blackwell | — (JIT) | `4` |

**GPU memory** decides three knobs, in this order:

1. `--max-model-len` (`MML`): the per-request prompt+generation budget you
   actually need — not the model's maximum. KV and several SFI buffers scale
   with it.
2. `VLLM_KV_CACHE_MEMORY_BYTES` (`KVB`): pin the KV pool explicitly whenever
   the GPU is shared or the model is large. **SFI's own state (compact KV,
   capture buffers, selector workspaces, ~2–3 GB at 8 slots) lives outside
   vLLM's `gpu-memory-utilization` budget**, so util-based auto-sizing
   over-allocates and OOMs.
   Budget: `weights + KV pool + SFI state < total VRAM`, and leave headroom
   for neighbors' peak usage on shared cards.
3. `batch size` / `max_live_sparse_slots`: compact slots are a *resident* KV
   cost — `slots × compact_blocks_per_slot × 16` tokens carved out of the KV
   pool. On small pools, shrink `slots` to your real batch and keep
   `compact_blocks_per_slot` near `k_head/16`.

In sparse mode the KV pool must therefore hold the **full KV of every request
plus the compact-page lease**. Capacity check (`kv_bytes/token` = layers ×
kv_heads × head_dim × 2 × dtype_bytes; 147456 for Qwen3-4B):

```
cap_tokens_per_req = (KVB − slots × blocks_per_slot × 16 × kv_bytes/token)
                     / (kv_bytes/token × batch)      must be  >  prompt + max_new
```

If it isn't, nothing crashes — the vLLM scheduler silently serializes the
batch: half the requests wait, prefill is recomputed mid-run, decode takes
~2× the steps and `decode_tps` roughly halves. `scripts/run_speed.sh` runs
this check automatically and warns before launching.

Validated A100-40GB tier examples are baked into `scripts/run_speed.sh`
(`bs8x12k` → 18 GiB KV + MML 16384, etc.) — scale them proportionally on
bigger/smaller cards.

**CUDA toolkit**: point `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` at any local
≥ 12.0 toolkit before `setup_flash_attention.sh`; nothing else reads it.

**Multi-GPU (tensor parallel)**: pass `--tensor-parallel-size N` (benchmark
driver and runners) with a matching multi-GPU `--cuda-visible-devices` list,
or `TP=N` for `scripts/run_speed.sh`. `VLLM_KV_CACHE_MEMORY_BYTES` is
per GPU. The runners apply two TP adaptations automatically:

1. vLLM's custom all-reduce is auto-probed: it stays enabled when every
   selected GPU has an active NVLink and is disabled on PCIe-only topologies,
   where the custom kernel kills worker init with
   `custom_all_reduce.cuh ... invalid argument`. Override with
   `--enable-custom-all-reduce` to force it on.
2. vLLM's async scheduling stays ON (it removes a fixed per-step host
   scheduling penalty that grows with TP size). Async feeds workers
   placeholder decode tokens; the SFI runtime harvests vLLM's own async
   sampled-token copy and repairs the placeholders in place, so the TP
   sentence-trigger token-source contract holds. `--sync-scheduling` (or
   `VLLM_SPARSE_SYNC_SCHEDULING=1`) forces synchronous scheduling back as an
   escape hatch.

Expectation management: on PCIe-only boxes the per-layer all-reduce dominates
small-batch decode for sparse and dense alike — use TP to *fit* longer
contexts (2 × 128k on two 40 GB cards), not to speed up decode.

<br>

## LongBench

1. Serve with quality settings:
   ```bash
   K_HEAD=4096 MML=65536 bash scripts/serve_sparse.sh 0 /path/to/model 8000
   ```
2. Clone the official [LongBench](https://github.com/THUDM/LongBench) harness,
   map your model in its `config/model2path.json`, then:
   ```bash
   LONGBENCH_ROOT=/path/to/LongBench bash scripts/run_longbench_v2.sh <MODEL_NAME> 4
   ```
3. `--n_proc` is the client concurrency (= decode batch pressure). To
   sub-sample, pre-write stub rows into the output file (see the script
   header); the harness's resume logic skips them.

<br>

## Troubleshooting

| Symptom | Meaning / fix |
|:--------|:--------------|
| `Free memory on device ... is less than desired` at engine start | another process grabbed the GPU — an environment failure, not an SFI bug; find a free GPU (`nvidia-smi`) |
| Engine-start OOM mentioning `Process XXX has N GiB` | shared card: pin `VLLM_KV_CACHE_MEMORY_BYTES` sized against your neighbors' **peak** usage |
| OOM only at large batch/context | lower `MML` to what you need, pin `KVB`, or drop a tier (`bs4x24k` → `bs2x30k`) |
| Sparse decode tps ~half of reference, decode step count ~2× expected, prefill re-runs mid-decode | KV pool can't hold full KV + compact-page lease for the whole batch → scheduler serializes it; raise `VLLM_KV_CACHE_MEMORY_BYTES` per the capacity formula above (`run_speed.sh` warns about this at launch) |
| Speed harness exits 2 with `unknown_without_reference` / `interval_trigger_intents_below_expected` | expected in free-corpus + `--skip-dense-reference` mode; judge by `decode_tps`, refresh counts, zero fallbacks (the script does this) |
| `Expected at least N 'Context:' segments` | your corpus has fewer segments than `--batch-size`; regenerate with `scripts/make_context_corpus.py` (never truncate a corpus with `head -c`) |
| No sparse routing in traces | check `/tmp/vllm_sparse_site.log` (`VLLM_SPARSE_SITE_LOG=1`): the controller JSON must parse, and `VLLM_SPARSE_FA3_UPSTREAM_ROOT` must contain the built `.so` |
| TP>1 worker dies at init with `custom_all_reduce.cuh ... invalid argument` | PCIe-only topology + vLLM custom all-reduce; the runners disable it automatically — don't pass `--enable-custom-all-reduce` on such boxes |
| TP>1 sparse raises `E_TP2_TOKEN_SOURCE_INCOMPLETE` | the worker token source is inconsistent: under synchronous scheduling this means placeholder tokens leaked in; "repair order corrupted" / "stash overflow" variants mean the async repair channel broke — rerun with `--sync-scheduling` and report it |
| After a TP run the GPUs stay occupied | sparse TP teardown can leave `VLLM::Worker_TP*` processes behind; find them via `nvidia-smi --query-compute-apps=pid ...` and kill before the next run |
| FA3 build fails immediately | `nvcc` not on `PATH` or CUDA < 12.0; set `CUDA_HOME` and retry |
| JIT helper-extension build fails | ensure `ninja` is installed and `TORCH_EXTENSIONS_DIR` is writable |
| Throughput far below reference | the GPU is not exclusive (any co-tenant skews decode timing), or thermal/power capped |
