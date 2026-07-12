# 远程测试指令（2026-07-13，selector 1301 + CHUNK18 本地终态）

> **状态分层**：当前已发布基线仍是 release `cuda-kernel` 的 **`a8456f5`**
>（selector semantic=`2026071201`）。其上的本地候选已升级至 semantic
> `2026071203`，但尚未产生新的 release commit；收到后续明确 release commit
> hash 前，不要拿 `a8456f5` 按 1203 二进制合同验收。新 commit 发出后，本档的
> 1203 段取代旧 1201 说明。Lite v1.1 仍未合；`a8456f5` 仍默认 chunk14，
> 本地 1203 候选已完成证据链并正式默认 chunk18。

## 2026-07-13 当前候选（覆盖下文 1203 旧口径）

> 当前代码已经在主树 AXP 与 `/ssd/xyxie/sfi-release-cuda` 逐文件对齐，但仍未
> `git commit`/`git push`。因此本节描述的是**可验收的本地候选**，不是新的远端
> commit；下文 1203 数据保留为演进历史，二进制合同以本节 1301 为准。

- selector semantic=`2026071301`，module/cache=
  `selector_pipeline_ext_v2026071301`。1301 把确定性 L2 与原 Python L1 正典化
  合成单一 CUDA producer；旧 1203 `.so` 不能复用，必须做一次 selector-only
  rebuild。发布树本机构建物 SHA256 为
  `995421642e5ff51ab2cf59d66b96a6f54c595d10dda162c658043387c9de694e`。
- FA gitlink、FA3 源码与 kernel patch 本轮仍为零变化，已持有正确 FA3 `.so` 的
  机器**不需要重编 FA3**。全新 checkout 仍须运行
  `scripts/setup_flash_attention.sh`；也可将
  `VLLM_SPARSE_FA3_UPSTREAM_ROOT` 指向已经准备好的 checkout。runner 会在 corpus
  生成和 selector JIT 之前验证 interface 与 `_vllm_fa3_C*.so`，缺失时立即
  `rc=66`，不再先白跑数分钟编译。
- runner 的 verdict-grade 条件现同时要求：独占规范化 GPU lock、sparse KV
  preflight、FA3 preflight、loader 实际 wire prompt 的精确 token corpus、fresh
  nonce、完整 8×256 `token_lengths`，以及 selector prewarm/postflight
  path/semantic/hash 完全一致。shared GPU、undersized KV override、旧 corpus 或
  缺失 FA checkout 均不能产生正式绿车。
- Qwen3-4B-Instruct-2507、A100 40GB、bs8×12k 的最终本地配对：sparse
  `decode_tps=268.9904`、`all_decode_tps=665.8281`；dense
  `175.9740`、`292.7292`，分别为 `1.5286x` 与 `2.2746x`。两车均为 8/8 请求、
  每请求 256/256 token、route/output/provenance 绿；sparse 仅保留允许的
  `unknown_without_reference` 噪声。
- 最终 release gate 为 `617 passed, 3 skipped`；另有 29 项红队定向契约和
  29 个生产文件双树逐字节一致性检查。Lite v1.1 仍停靠未合，其专属测试不属于
  本候选 gate。

推荐配方：

```bash
PYTHON=/abs/path/to/vllm-python \
VLLM_SPARSE_FA3_UPSTREAM_ROOT=/abs/path/to/prepared/flash-attention \
bash scripts/run_speed.sh "0" /abs/path/to/model bs8x12k sparse <unique-tag>
```

## 本次包含

- P1-B 世代级 gap 节流与 sentence/interval 交互合同。
- L2 selector top-k 并列决定论，以及定制 CUDA post-topk 稳定压紧。
- selector extension 语义版本 `2026071203`，module/cache name=
  `selector_pipeline_ext_v2026071203`；旧 1201/1202 `.so` 不再可能同名混载。
- `_CAPTURE_CHUNK` 默认 14→18；环境变量仍可显式设 14 回滚。
- 合法 `last_decode_refresh_step=0` 保留，不再把 step 0 当“未初始化”。
- 三处 persistent pinned-host source 的 non-blocking H2D 生命周期闭合。
- `target_layer_start=0` 的 deadline 排序修复。
- `_interval_merge_policy="off"` 兼容档 mixed bus 的排列不变性修复；默认
  `delta1` 路径保持原策略。

## 二进制义务

1. **FA3 本增量不变**：若机器已经基于 `3325c0d` 重编并验证过 FA3，拉
   `a8456f5` 或后续 1203 selector-only 增量后均无需再次重编 FA3 `.so`。全新
   安装或旧于 `3325c0d` 的机器仍按
   `scripts/setup_flash_attention.sh` 正常构建。
2. **selector_pipeline_ext 必须重建**：建议使用新的私有
   `TORCH_EXTENSIONS_DIR`。1203 首跑只需一次 selector JIT；运行时会检查
   semantic/entrypoint，扩展名本身也带语义版本。不要复制/重命名旧 `.so`
   绕过合同。
3. FA4/SM100 patch 叠加链仍未完成 rebase；本次只声明 SM80/SM90 release-ready。

旧实现的根因不是“磁盘 `.so` 没覆盖”：同一 Python 进程已经 dlopen 某 module
name 后，即使原路径被重编为新语义，loader 仍可能返回旧映像。把 semantic 写入
module/cache name 才能彻底隔离。因此本次是**一次 selector-only rebuild**，不是
FA3 rebuild。

## 判据锚（按 commit/default 分层）

- **1203 候选、默认 chunk18**：黄金 bs2
  `{ad585b37,44e80946}`；bs8x12k 8-hash
  `{970c061e,c0a6376d,99a9aea2,03eb3325,fda56972,deef29a8,f6d008c1,4076e3d5}`；
  counts `{sentence:16,interval:16}`、payloads `648`。本地 full all-decode TPS
  `670.0263`。
- **`a8456f5`/显式 chunk14 回滚锚**：bs8x12k 8-hash
  `{fff0445e, c0a6376d, 99a9aea2, 7db6f96d, fda56972, deef29a8, 8e8e2dca, 8778b158}`。
  counts `{sentence:19,interval:16}`、payloads `756`；根修后 full all-decode TPS
  `662.2989`。远端绝对值以机器实测为准。

golden c14/c18 均已复验：`gate_passed=true`、`production_gate_passed=true`、
hash `{ad585b37,44e80946}`、counts `{sentence:2,interval:2}`、payloads `84`。

旧锚 `{636fb032,44e80946}` 与旧 counts `{sentence:15,interval:16}/612` 不再
用于最终 release 验收。严格单变量 A/B 已证明：旧代码把合法
`last_decode_refresh_step=0` 解析为 `-1`，从而绕过 min-gap，让首个 producer
work 在 step 4 提前发生；修复后为 step 16，后续 work 从 `[4,99,100]` 变为
`[16,111,112]`。仅回退该零值修复即可逐位恢复旧黄金锚，因此本次换锚有明确
因果，不是 run 间随机漂移。

## 配方

黄金 bs2：

```bash
python benchmarks/bench_sm80_mixed_page_one_shot_graph_e2e.py \
  --output out/<tag>.json --summary-output out/<tag>_summary.json \
  --mode sparse --preset bs2long-cap128 --producer-mode full-open-gt1 \
  --model /path/to/qwen3-0.6b --python /path/to/vllm-python \
  --cuda-visible-devices <GPU> --skip-dense-reference --outputs-include-text
```

判速 bs8x12k：

```bash
VERDICT_ONLY=1 PYTHON=/path/to/vllm-python \
  TORCH_EXTENSIONS_DIR=/path/to/private/selector-1203-cache \
  bash scripts/run_speed.sh <GPU> /path/to/Qwen3-4B-Instruct-2507 \
  bs8x12k sparse <unique-tag>
```

`run_speed.sh` 对每个 TAG 持有非阻塞 `flock`；并行任务必须用不同 TAG。它会保留
harness rc，并由 postflight 检查本次 summary 的新鲜度、route/lifecycle/
producer/reference/production gate，以及 selector prewarm/postflight 的
path+sha256+semantic。`VERDICT_ONLY=1` 下可接受的 sparse no-reference 噪声只有
以下两个**完整字符串**：

- `semantic_output_health_not_ok:unknown_without_reference`
- `interval_trigger_intents_below_expected`

`async_producer_writer_missing` 是真实 producer failure，不再允许作为裁剪噪声。
default18 终复验已完成，`run_provenance.capture_chunk_effective=18`；VO route
writer-complete count=36，与 full refresh-profile writer count=36 精确一致，
producer reasons 为空，`SPEED RUN OK`。该 count 是真实 writer-complete
measurement，不是 allowlist；不得把 `async_producer_writer_missing` 加入
allowlist。首轮远端仍建议补一发非 VERDICT_ONLY 全形态作为对照。

## 最小验收

- 黄金：`gate_passed=true`、`production_gate_passed=true`、
  `route=ResolvedRowPtr`、counts `{sentence:2,interval:2}`、payloads `84`。
- bs8x12k：同配置至少三发 8-hash、counts、payloads 逐位互同。
- selector：actual/expected semantic 均为 `2026071203`；prewarm/postflight
  path、sha256 相同，`selector_pipeline_artifact_mutated_after_prewarm=false`。
  本地证据为相关 selector 87 passed、整合定向 89 passed；覆盖真实 CUDA、
  CUDA graph、±inf 与 NaN fail-loud。最终 runner/capture/kernel/producer/selector/
  one-shot 合同并集为 **436 passed**，其中 default18、post-bridge min-gap 与
  writer-default 单真源测试债已收口。default18 的 262K arena 预算现由
  capture chunk/in-flight 单真源推导（当前向上取整为 5 GiB），live
  `seq_lens_batch_i32` 也在预备阶段完整落位，refresh cache-hit 不再补建。
- kernel-only 负控必须使用 production RRP pinned 三 case 与
  `--force-num-splits 1`；本地结果 gate=true、max overhead=0.215745%≤0.4%。
  不接受默认 broad `non_tma` 的证据空集 green。broad split1 的
  direct_physical=0.428263% 红腿如实记录，但不是 production RRP 阻断门。
  多 cell 聚合按 `(batch,requested_kv_len,case)` 逐格 fail closed，禁止用一格
  matched 掩盖另一格 skipped。
- 若锚不符，先核对 `run_provenance.git_head`、selector semantic version、
  `TORCH_EXTENSIONS_DIR` 和 FA upstream 路径，禁止直接改锚掩盖环境漂移。

## 本地优化数据与当前默认

- TP8 代理：selector 总耗时 11.038→6.071 ms（1.818x），post-topk
  7.239→1.847 ms（3.92x），临时显存 766.31→37.45 MiB（−95.1%），checksum
  相同。
- capture scratch full 4B 已发现 `[2,256,32,2,16384]` fp16=1 GiB：benchmark
  child 漏传 `max_num_seqs`，沿用 vLLM 默认 256。这是真实过配；本地命令链已
  pin `max_num_seqs=batch_size=8`，目标 shape `[2,8,32,2,16384]`=32 MiB
  （32x/−96.875%）；针对性 12 passed。full 4B c14/c18 已实测 prebuild/live
  均为 32 MiB 且 live `cache_hit=true`。**release 同步完成前，不发 1203 远端
  候选。**
- PBD 现实可回收 0.408%，绝对上界 0.736%，低于 1% gate，停止开发。
- CHUNK18 在 4B 六腿均值 +1.0156%，但 counts/payload 与 4/8 输出 hash 改变。
  golden c14/c18 与 full 两形态现已按各自锚闭环，故 1203 候选**正式默认 18**；
  远端验收必须使用上节 c18 新锚。需要问题二分时可显式设
  `VLLM_SPARSE_CAPTURE_CHUNK=14`，并按 c14 回滚锚判读，不能混用两套 counts/hash。

kernel-only pinned 配方：

```bash
python benchmarks/bench_mixed_page_resolver_kernel_only.py \
  --case-filter kind0_native_non_tma_page64,kind4_rowptr_batch_row_non_tma,kind4_rowptr_irregular_stride1_non_tma \
  --force-num-splits 1 --output <json> --fail-on-gate
```
