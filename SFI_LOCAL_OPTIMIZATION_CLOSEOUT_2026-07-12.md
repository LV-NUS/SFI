# SFI 本地优化收针（2026-07-12，release 合仓后的继续优化）

> 本档晚于 `SFI_MERGE_GUIDE_2026-07-12.md` §0.6 与
> `SFI_NIGHT_SESSION_HANDOFF_2026-07-11.md` §13.6。它记录的是
> **release `a8456f5` 之上的本地候选工作树**，不是新的已发布 commit。
> 在完成黄金/全形态复验、同步 release 并取得 commit/push 授权前，远端生产锚仍是
> `a8456f5`。

## 0. 2026-07-13 最终 infra closeout（覆盖后文 1203 中间态）

- kernel/host 代码已经在完整源树与 release `cuda-kernel` 工作树对齐；29 个生产
  文件逐字节相同，Python 静态编译、shell 语法、隔离 release CLI 与
  `git diff --check` 全绿。尚未 commit/push。
- selector semantic 从 1203 升至 `2026071301`：确定性 L2 CUDA kernel 直接写
  int32 正典 pack order，删除中间 i64/f32、i64→i32 kernel 与 Python L1
  masked-fill/sort 链。旧 1203+L1 `1.977344 ms`，1301 fused
  `1.901568 ms`（约 +3.8%）；10/10 对拍逐位一致，39 项 selector GPU/CPU 合同
  全绿。发布树 `.so` SHA256=
  `995421642e5ff51ab2cf59d66b96a6f54c595d10dda162c658043387c9de694e`。
- prefill prepared-only step/chunk memo 将 36 层 layout lookup/bind/lease 从
  `36/36/36` 降至 `2/2/2`，并把已解析 layout 直接传给 payload；fallback 与
  live-length 缺失路径禁止 memo，避免用性能缓存掩盖状态漂移。
- forward capture 缓存 absent-phase int32 mapping，36 层分配 `36→1` 且指针在
  CUDAGraph replay 下稳定；强 CPU truth 已验证时移除每层 6 个 device assert，
  共少 216 个小 kernel/assert，弱证据路径仍保持 fail-closed device 校验。
- StepBoundMeta 只有存在 authority 且完成 coverage/hash 验证后才发布 memo；
  authority 缺失永不 memo，后到 hash 漂移仍会失败。corpus cache 升至 schema
  v2，按 loader 实际 `Context:` wire prompt 保证精确长度；Qwen 8×12k cache
  miss `4.83 s/RSS 823112 KB`，hit `0.06 s/RSS 19284 KB`，实际长度均为 12000。
- runner 新增 ABI 分区 selector cache、规范化 GPU/TAG lock、sparse KV 与 FA3
  checkout/build fail-fast、精确 corpus validator、nonce/artifact/token-length
  postflight。首次 release 实车由此暴露“缺 vendored FA 时 sparse 先白编 JIT”
  的设计缺口；现在缺 interface/`.so` 在 harness 前 `rc=66`，并给出 setup/env
  修复命令。
- 真实 Qwen3-4B-Instruct-2507 bs8×12k 配对最终为：sparse
  `268.9904 decode TPS / 665.8281 all-decode TPS`，dense
  `175.9740 / 292.7292`；提升分别为 `1.5286x` 与 `2.2746x`。两车均独占 GPU、
  精确 8×12k 输入和 8×256 完整输出，FA3 hash=
  `fc8dbbcf8eaa1baab0412e31000f087af911409d5bb9066e2e40acc186b5e96c`。
- 最终统一 release gate `617 passed, 3 skipped`；红队复验 29 项全绿。另扫到的
  15 个 Lite v1.1 专属失败来自明确停靠未合分支，生产树没有其函数，不作为本次
  gate，也没有为迎合陈旧合同修改生产代码。

二进制结论不变但版本更新：本轮**必须重建 selector 1301**；FA3 源/gitlink/
patch 未变，因此已有正确 FA3 基座无需重编。全新 release checkout 缺 vendored
FA 时仍须正常 setup，或显式指向已准备 checkout；这属于依赖准备，不是本轮
kernel 变更。CHUNK18 保持默认，`VLLM_SPARSE_CAPTURE_CHUNK=14` 仍是回滚开关。

## 1. 结论

- kernel 与 host 已在 release `a8456f5` 合仓；其 FA3 kernel 基座仍是
  `3325c0d`。本轮本地继续优化没有改 FA gitlink、FA3 源码或 kernel patch，
  **不触发 FA3 重编**。
- L2 已从高临时显存的 ATen 组合改为定制 CUDA post-topk 稳定压紧；selector
  语义版本升为 `2026071203`，扩展名/缓存键改为
  `selector_pipeline_ext_v2026071203`。升级后只需一次 selector JIT/prebuilt
  重建，不能复用语义 `2026071201` 的旧 selector `.so`。
- runner 与 selector artifact provenance 已本地落地；capture scratch 探针发现
  benchmark 的真实 1 GiB 过配，max_num_seqs 传递根修已落并完成 full 4B 复验。
  PBD 因可回收收益低于 1% 停止线而 no-go；CHUNK18 在 4B 六腿上有约 1.02%
  收益；黄金与 full 两档证据闭环后，本地候选已正式把全局默认从 14 升为 18。

## 2. L2 定制 CUDA 结果

实现位于 `release/attention-xx-point/utils/selector_pipeline_ext.py`：先从 O(k)
top-k 样本计算严格大于边界的数量，再按原列序扫描 width，稳定发射边界并列项。
输出先写安全哨兵；NaN 或最终发射数量不等于 k 时 device assert，禁止返回未初始化
数据。

TP8 代理微基准（固定输入、同 checksum）：

| 指标 | 旧 ATen 组合 | `2026071203` | 变化 |
|---|---:|---:|---:|
| selector 总耗时 | 11.038 ms | 6.071 ms | 1.818x，−45.0% |
| post-topk | 7.239 ms | 1.847 ms | 3.92x |
| 峰值临时显存 | 766.31 MiB | 37.45 MiB | −95.1% |
| 新总耗时/raw top-k | — | 1.496x | 达到 ≤1.5x 门 |

本地构建物：

- path：`/tmp/sfi_l2_twopass_1203_20260712/selector_pipeline_ext_v2026071203/selector_pipeline_ext_v2026071203.so`
- sha256：`ff35b94107c49ce011f13c97e1e627b6a8dba0fe1ea838bd2b9efe867fc3f4fc`
- 两个独立 child 直载约 1.60/1.57 ms，path/hash/mtime 一致。
- 相关 selector 套件 87 passed；runner+capture+selector 整合定向套件
  89 passed。覆盖 no-tie、all-equal、k==width、slice、padding/sentinel、±inf、
  CUDA graph，以及 NaN 隔离子进程 fail-loud。

### 2.1 stale dlopen 与重复 JIT 根因

旧实现即使在同一路径把同名 `.so` 从语义 1201 重编成 1202，当前 Python 进程的
dynamic loader 仍会返回已加载的旧 1201；磁盘 hash 变化并不能证明进程内代码已
切换。处理如下：

1. 扩展名和缓存目录纳入语义版本：
   `selector_pipeline_ext_v2026071203`。
2. prebuilt 与 JIT 返回对象都再次校验 semantic/entrypoint；不匹配 fail closed。
3. selector 侧固定 `TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES=1`，消除不同
   import 顺序交替改写 Ninja 命令而导致的每腿约 48 秒重复重编。

因此远端升级的二进制义务是**一次 selector-only rebuild**；FA3 无本轮 rebuild
义务。

## 3. runner 与 artifact 合同

`scripts/run_speed.sh` 不再吞 harness return code，并由
`scripts/check_run_speed_summary.py` 对本次新鲜 summary 做 fail-closed postflight：

- selector prewarm/postflight 必须同时记录实际 path、sha256、actual/expected
  semantic；两端或 `run_provenance` 任一缺失、不一致、文件被改写，整车失败。
- speed/diagnostic child rc、timeout、fatal，内外 route proof、workload replay、
  lifecycle、producer/reference/production gate 任一不绿，均不能用 rc=2 掩盖。
- sparse no-reference 的精确 allowlist 只有：
  `semantic_output_health_not_ok:unknown_without_reference` 与
  `interval_trigger_intents_below_expected`。按完整字符串比较，不接受 substring/
  suffix 碰撞。
- `async_producer_writer_missing` 是真实 producer gate failure；即使
  `VERDICT_ONLY=1` 也不得当作裁剪噪声。
- 每个 TAG 持有非阻塞 `flock`。系统无 flock 时 rc=69；同 TAG 已在跑时 rc=75。
  并行发车必须用不同 TAG。

default18 runner 的两项收口已完成并复验：

1. default18 的真实输出/arena 已是 c18，但 `run_provenance` 曾把默认值硬编码为
   14；现已改为从 `sparse_constants` 单真源读取，终复验记录 18。
2. default18 的 VERDICT_ONLY speed child 只有 hook enqueue 证据，没有 writer
   invocation/complete 证据时曾被 `async_producer_writer_missing` 正确 fail closed。
   现已加入轻量 route writer-complete measurement：VO count=36，与 full
   refresh-profile writer count=36 精确一致，producer reasons 为空。

终复验 tag=`local_l2_1203_default18_vo_writerproof`：c18 8/8、counts 16/16、payload
648、all-decode TPS 668.7951、provenance chunk=18，selector 1203 hash 前后同，
`SPEED RUN OK`；唯一 gate noise 是精确允许的 `unknown_without_reference`。
writer route evidence 是**真实 measurement**，不是 allowlist 扩张；
`async_producer_writer_missing` 仍为真红。

验收时除 hash/counts/TPS 外，必须检查：

```text
selector_pipeline_artifact_prewarm
selector_pipeline_artifact_postflight
selector_pipeline_artifact_mutated_after_prewarm == false
actual semantic == expected semantic == 2026071203
prewarm sha256 == postflight sha256
```

## 4. capture scratch、PBD 与 CHUNK18

### 4.0 kernel-only 负控

旧 broad `--case-filter non_tma` 在默认形态可能把 production RRP 对全部 skip，旧
aggregate 因证据空集而 vacuous green；现在 aggregate 已改为无 RRP 证据即
fail closed。后续审计又补齐多 cell 隔离：以 `(batch, requested_kv_len, case)`
聚合 missing/duplicate 与 launch/speed evidence，逐 record 检查 RRP；一个 cell
matched 不能掩盖另一 cell skipped，同名 case 跨 cell 也不再误报 duplicate。该根修
合同 25 passed。本轮正确的 production RRP pinned 配方是：

```bash
python benchmarks/bench_mixed_page_resolver_kernel_only.py \
  --case-filter kind0_native_non_tma_page64,kind4_rowptr_batch_row_non_tma,kind4_rowptr_irregular_stride1_non_tma \
  --force-num-splits 1 --output <json> --fail-on-gate
```

实测 gate=true，max RRP overhead=0.215745%（门 ≤0.4%）。broad split1 的三次
direct_physical=0.428263% 而红，说明该非 production 形态确有噪声/税；如实保留，
但它不属于 production RRP selector merge gate，不能替代也不能否决 pinned 负控。

### 4.1 capture scratch

新增默认关闭的 `VLLM_SPARSE_CAPTURE_SCRATCH_PROBE_LOG=<jsonl>`。探针按
source+allocation key 去重，并在 JSON 序列化/文件锁前返回，启用时也不会每层每步
重复做 I/O。full 4B/bs8x12k 首次实测 shape 是
`[2,256,32,2,16384]`、fp16=1,073,741,824 B（1 GiB）：E2E child 漏传
`max_num_seqs`，vLLM 沿用了默认 256。这是当前真实 over-allocation，不得再解释
为旧 MML shape。代码已让 bs8 命令链显式 pin `max_num_seqs=8`，目标 shape
`[2,8,32,2,16384]`=33,554,432 B（32 MiB），理论缩减 32x/−96.875%，live 应
`cache_hit=true`；针对性测试 12 passed，py_compile/diff-check 绿。根修后 full
4B 的 chunk14/chunk18 两车均实测 prebuild/live shape=目标 32 MiB，且 live
`cache_hit=true`，故 32x 显存回收已闭环。

### 4.2 PBD no-go

历史“38 miss”不是 38 个关键路径 step。判速 speed child 中唯一可外推 PBD miss
只有 8 个：现实可回收 12.189 ms / 2.9848 s = 0.408%；把全部成本理想化清零的
绝对上界也只有 21.968 ms = 0.736%。二者均低于项目 1% gate，现实值也低于约
0.5% stop rule，故不投入约 1500 行 PBD arm。

### 4.3 CHUNK18 正式转全局默认 18

4B、release `a8456f5`、同卡 VERDICT_ONLY 六腿 A/B：

- chunk14：656.6315 / 659.0258 / 658.4105 TPS，均值 658.0226。
- chunk18：665.6358 / 663.7315 / 664.7490 TPS，均值 664.7054。
- 均值 +6.6828 TPS / +1.0156%；三组 paired delta 均为正。
- refresh 平均耗时 8228.31 → 7614.91 us（−7.45%）。
- 但 counts 从 19/16、payload 756 变为 16/16、payload 648，且 8 个输出中
  4 个 hash 改变。
- 根修后的 full 单对照同向：chunk14=662.2989、chunk18=670.0263 all-decode TPS
  （+1.1668%）；两车 selector semantic/hash 前后稳定，只有精确允许的
  `unknown_without_reference` gate noise。

结论：这是有性能信号的**语义/触发节奏变更**，不是可静默替换的纯实现优化。
本轮没有把它伪装成透明优化，而是完成了对应的新锚流程：

- golden c14/c18 均 `gate_passed=true`、`production_gate_passed=true`，hash
  `{ad585b37,44e80946}`、counts 2/2、payload 84。
- full c14 保持回滚锚
  `{fff0445e,c0a6376d,99a9aea2,7db6f96d,fda56972,deef29a8,8e8e2dca,8778b158}`，
  counts 19/16、payload 756、all-decode TPS 662.2989。
- full c18 命中新的 production 锚
  `{970c061e,c0a6376d,99a9aea2,03eb3325,fda56972,deef29a8,f6d008c1,4076e3d5}`，
  counts 16/16、payload 648、all-decode TPS 670.0263。

因此 `patches/sparse_constants.py` 的 `_CAPTURE_CHUNK` 默认已从 14 正式升为 18，
并新增 promotion 注释/默认测试。环境变量仍可显式设 14 回滚；从本地 1203 候选
开始，4B production semantic/counts/hash 必须按 c18 新锚验收。实际 default18
输出/arena、provenance 单真源与 VO writer-complete 均已复验闭环。

### 4.4 最终合同并集与测试债收口

最终把 runner、capture、kernel-only、producer、selector、one-shot 与 arena 合同
合并在同一 pytest 进程复跑，结果为 **436 passed**。原 417-case 合并腿额外暴露
并收口了五个
陈旧测试假设：chunk14 的 15-layer/ready7 固定值、post-bridge 忽略 min-gap、
repo 位于 `/ssd` 时的自相矛盾路径断言，以及 writer tile 仍期待旧默认 16。
测试现从 capture/writer 单真源派生；post-bridge 同时覆盖 `min_gap=0` 的下一步
追赶与默认 gap 内保留、gap 到点才触发两条合同。生产 writer 默认也提取为
`_WRITER_TOKEN_TILE_DEFAULT=128`，provenance 与运行时共用，避免再次发生硬编码漂移。

追加的 arena 审查还收口了两个真实集成缺口：固定 4 GiB 的 262K 预算只覆盖
chunk14，default18 的双缓冲实际投影约 4.50 GiB，会在分配前直接拒绝；预算现从
`_CAPTURE_CHUNK`/`_CAPTURE_IN_FLIGHT` 推导并向上取整到 5 GiB。预备阶段也同步
写入 `seq_lens_batch_i32`，避免 refresh cache-hit 热路径额外重建。

## 5. 接续顺序

1. `max_num_seqs=batch_size` 根修与 full 4B 32 MiB/cached-hit 证据已完成；保留
   chunk14 正典 hash/counts 作为后续回归锚。
2. 黄金 c14/c18、非 VO full c14/c18 与 §4.0 pinned kernel-only 均已完成；后续回归不得
   退回 vacuous broad 口径，且须继续核对 selector artifact path/hash 前后不变。
3. 将已验证的生产文件同步 release 工作树，逐文件 sha256/cmp；更新 release commit
   hash 后再发远端指令。
4. commit/push 属高风险操作，必须另取用户明确确认；本档本身不代表已提交或已发布。
