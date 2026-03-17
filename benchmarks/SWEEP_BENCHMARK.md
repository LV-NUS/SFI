# Decode Throughput Sweep Benchmark

`run_sweep.py` sweeps decode throughput across multiple context lengths in a single model load, supporting both dense and sparse modes.

## Quick Start

```bash
# Dense mode: sweep 8k / 16k / 32k
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  python benchmarks/run_sweep.py --mode dense \
    --model <MODEL_PATH> \
    --context-lengths 8192,16384,32768

# Sparse mode: sweep 8k / 16k / 32k
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  python benchmarks/run_sweep.py --mode sparse \
    --model <MODEL_PATH> \
    --context-lengths 8192,16384,32768
```

## Core Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | (required) | `dense` or `sparse` |
| `--model` | `./qwen3-0.6b` | Model path |
| `--batch-size` | 4 | Concurrent request count |
| `--max-new-tokens` | 2048 | Tokens to generate per request |
| `--context-lengths` | `8192,16384,32768,65536,131072` | Comma-separated context lengths to sweep |
| `--warmup-runs` | 1 | Warmup rounds per context length |
| `--measure-runs` | 3 | Measurement rounds per context length |
| `--gpu-mem-util` | 0.9 | GPU memory utilization fraction |
| `--dtype` | `bfloat16` | Data type (`bfloat16` or `float16`) |
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism |
| `--output-json` | `benchmarks/sweep_{mode}_results.json` | Output path for results |
| `--temperature` | 0.0 | Sampling temperature (0.0 = greedy) |
| `--enforce-eager` | off | Disable torch.compile |
| `--disable-prefix-caching` | off | Disable automatic prefix caching |
| `--max-seq-len-to-capture` | 32768 | Max sequence length for CUDA graph capture |
| `--disable-cascade-attn` | True | Disable vLLM V1 cascade attention |
| `--enable-cascade-attn` | — | Enable cascade attention (overrides above) |

## Sparse-specific Arguments

These arguments only take effect when `--mode sparse`. The defaults are aligned with common configurations and typically do not need to be changed.

| Argument | Default | Description |
|----------|---------|-------------|
| `--alpha-k-head` | 1536 | Tokens retained per attention head |
| `--tau` | 1.0 | Selection temperature |
| `--k-min` | 32 | Minimum retained tokens |
| `--k-max` | `none` | Maximum retained tokens (`none` = unlimited) |
| `--sink` | 4 | Number of fixed sink tokens (always retained) |
| `--recent` | 256 | Number of recent tokens (always retained) |
| `--refresh-interval` | 32 | Refresh the selector every N decode steps |
| `--min-refresh-gap` | 16 | Minimum gap between refreshes in decode steps |
| `--refresh-layer-groups` | 2 | Layer-group gating for refresh |
| `--prefill-last-n` | 16 | Prefill capture window (<=0 disables capture) |
| `--disable-sentence-trigger` | True | Disable sentence-based refresh triggers |
| `--enable-sentence-trigger` | — | Enable sentence-based refresh triggers |
| `--selector-fixed-k` | True | Set `VLLM_SPARSE_SELECTOR_FIXED_K=1` |
| `--no-selector-fixed-k` | — | Unset `VLLM_SPARSE_SELECTOR_FIXED_K` |
| `--gather-autotune` | `0` | Override `VLLM_SPARSE_GATHER_AUTOTUNE` |
| `--wait-policy` | `split` | Set `VLLM_SPARSE_WAIT_POLICY` (`chunk` or `split`) |
| `--capture-kv-bucket` | 2048 | Override `VLLM_SPARSE_CAPTURE_KV_BUCKET` |

## Steady-state Decode (No Refresh)

To measure the upper-bound decode throughput without refresh overhead:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  python benchmarks/run_sweep.py --mode sparse \
    --model <MODEL_PATH> \
    --context-lengths 8192,16384,32768 \
    --refresh-interval 1000000000 \
    --disable-sentence-trigger
```

> This disables refresh entirely. Token selection is never updated, so this is only suitable for performance upper-bound measurement, not quality evaluation.

## Measurement Methodology

```
Load model (once)
  │
  ▼
For each context length:
  ├── Build prompts (tile text to target token count)
  ├── Warmup N rounds (exclude compilation / CUDA graph effects)
  ├── Measure M rounds (step-by-step decode timing)
  └── Aggregate results (mean ± std)
  │
  ▼
Output summary table + JSON file
```

Key design decisions:
- **Independent warmup**: Each context length warms up independently, since new lengths may trigger `torch.compile` recompilation
- **Prefix cache reset**: Reset between rounds to avoid cache interference
- **Precise decode timing (TP-comparable)**: Uses `EngineCoreOutputs.timestamp` instead of outer wall time for accurate decode window measurement

### Throughput Metrics

**Primary metric — All-Decode Window (recommended):**

When the batch contains multiple requests with different prompt lengths, chunked prefill causes them to enter decode at different steps. During this transition, only a subset of requests produce decode tokens per step, and step latency is dominated by prefill computation (600–2000 ms/step). Including these steps in the decode window severely underestimates true decode performance.

The All-Decode window starts timing from the point where **all requests are in decode**:
- `t_ad_start`: First `EngineCoreOutputs` timestamp where `new_tokens >= batch_size`
- `t_end`: Last timestamp with new tokens
- `ad_tokens`: All tokens produced after `t_ad_start`
- `all_decode_tok_s = ad_tokens / (t_end - t_ad_start)`

**Secondary metric — First-Emit Window:**
- `t_start`: First timestamp with new tokens
- `t_end`: Last timestamp with new tokens
- `window_tokens`: Total generated tokens minus first-emit tokens
- `first_emit_tok_s = window_tokens / (t_end - t_start)`

The summary table reports `AllDecode tok/s` as the primary metric and `FirstEmit tok/s` as a secondary reference. When `batch_size=1` or all prompts are the same length, the two metrics are equivalent.

## Output Example

```
========================================================================
Summary  |  Mode: dense  |  Batch: 4  |  Max New Tokens: 2048
========================================================================
Context    AllDecode tok/s (mean+/-std)   FirstEmit tok/s   Prefill(s)   Total(s)  Warmup(s)
--------------------------------------------------------------------------------------------
8k                  512.34 +/- 2.10            508.20        0.432     16.112     18.234
16k                 478.90 +/- 1.85            462.10        0.891     17.234     20.456
32k                 404.31 +/- 3.20            388.50        1.729     21.980     24.357
```

Results are also written to a JSON file (default: `benchmarks/sweep_{mode}_results.json`), where `all_decode_tok_s_mean` is the primary throughput metric.

## GPU Memory Estimation

KV cache memory per token: `2 × num_layers × num_kv_heads × head_dim × 2 bytes`

| Model | KV Cache per Token | batch=4 × 128k |
|-------|-------------------|----------------|
| Qwen3-0.6B (28 layers, 8 heads, dim=128) | 112 KB | ~55 GB |
| Qwen3-4B (36 layers, 8 heads, dim=128) | 144 KB | ~70 GB |

If you encounter OOM:
1. Reduce `--batch-size` (minimum 1)
2. Reduce `--max-new-tokens`
3. Use GPUs with more memory
