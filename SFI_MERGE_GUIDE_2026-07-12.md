# SFI 停靠分支合流指引（kernel 侧+host 侧优化 merge 手册）

> 2026-07-12 傍晚定稿。**自包含**：新 session/agent 只读本档即可执行合流，不需先翻其它交接档。
> 写作时 main=`6c8f4a0`（若已前进，本档的"main 侧后续改动"清单需用 §5.0 的命令重新实查一遍再动手）。
> 主仓=/ssd/xyxie/sfi-fa-refresh-runtime/release_version（下称主树；无 remote=本地留档）；
> bench 根=主树/release/attention-xx-point（下称 AXP）；
> 发布仓 worktree=/ssd/xyxie/sfi-release-cuda（remote=LV-NUS/SFI，分支 cuda-kernel，最终合仓提交 `a8456f5`）。
> 解释器（一切 bench/pytest 必用绝对路径）：`/ssd/xyxie/miniconda3/envs/vllm019-cu126/bin/python`。

---

## §0 三十秒定位

- **已在 main、无需再合**：kernel 场四窗全部（SPLIT-ROOT 家族/K1/K2/N1/K6/N3/N4-reopen/P-2 lse arena/oracle 同 s 化/ext 九 patch `e234a33`/triton 终下线 `a703cc9`/21 预存红追平 `ccbf85e`/irregular 终谳 `3a0ff14`）+host 场合入窗五项（pack-cuda `3823c50`/小件批 `b7c6465`/CHUNK18 战报 `a2a9cec`/#14 u14 `32b1015`/ext 增量终局 `7458eef`）。
- **真正待合的只有三支**（全部停靠在旧基座 `c56ed44` 上，均为 host 场施工）：
  1. `p1b-gap-throttle` @`e077c70`（P1-B 世代 gap 班车节流，方向 B）——换锚件
  2. `l2-topk-tie` @`27cdb87`（L2 selector topk 并列决定论两段法）——换锚件+C++ ext 实变
  3. `lite-v11-arm-body` @`cb9178e`（Lite v1.1 SIG_RETURN 返程臂，四 commit 链）——**拍板件**（推荐停靠不合，见 §8）
- **合流的正确形态=一个"换锚窗"**：三支（或拍板后的子集）rebase→逐支判据→组合总闸→换锚腿 ×3 自证→呈新锚拍板→发布仓同步。全流程见 §3。
- **★写作时点换锚窗已由 host 场实际开启**：验证树分支 `anchor-window` @`e8f41ae`（=main+P1-B+L2 预合，worktree 在 `.claude/worktrees/anchor-window`），判据腿在跑，结果将回填主交接档 §13.3。**接续者第一动作=看 §13.3 是否已回填**：已回填且绿 ⇒ 直接从 §3 第 5 步（呈拍板）继续，前四步勿重做；未回填 ⇒ 查 anchor-window worktree 与 GPU 占用判断是否还在跑/夭折，再决定接续或重跑。
- 动手前必读：主交接档 `SFI_NIGHT_SESSION_HANDOFF_2026-07-11.md` 的 **§13**（host 场收工终账+其分支处置表——与本档互补：本档=全局自包含手册含判据配方/发布仓/协作纪律全量，§13=host 场视角+换锚窗最新进展；细节若有出入以更晚写入者+实查为准）、本档 §5.0（重新实查 main 侧漂移）、§6（多 session 协作纪律）。

## §0.5 ★★状态刷新（07-12 晚，host 场收工写入——本节晚于 §0/§1/§3 写作时点，以本节为准）

- **换锚窗已收针，结果=判据面零翻锚**：anchor-window 组合腿判速 ×3（8-hash 现锚 **8/8 逐位 ×3**+run 间互同+counts {sentence:15,interval:16}/612 精确）+黄金 bs2 **{636fb032,44e80946} 逐位+gate True**。两支行为差只在非判据形态（P1-B→gap≥24 生产/gap48 相变；L2→生产大尺度并列），判据档位（bench gap16+bs2 黄金）行为逐位不变。**⇒ §3 第 4/5 步的"候选新锚拍板"自动消解（锚全保持），§8 拍板件①（新锚转正）除名。**
- **P1-B+L2 已合 main**（`754c9c3`/`e2f5252`，与 anchor-window 树代码逐位等价=判据直接携带）。**待合只剩 `lite-v11-arm-body`（拍板件，推荐停靠）**——§0/§1.1 的"三支"表述过时。
- `anchor-window` 分支/worktree=已收编，可清（§1.2 的"勿删勿动"过时）。
- **发布仓义务 delta 追加**：P1-B（vllm_sparse_patch/sentence_triggers/interplay 测试）+L2（selector_pipeline_ext.py **C++ 第二次实变**——与刀A 叠加后以 main 终形态重打该 ext prebuilt+determinism 测试）。
- **新坑三条（worktree 黄金腿血账，已实锤）**：
  1. **历史 worktree FA 路径坑已根修**：当前 `get_vendored_upstream_root()` 会优先读取 `VLLM_SPARSE_FA3_UPSTREAM_ROOT`，不再强制依赖 worktree 的空 gitlink。`run_speed.sh` 还会在 corpus/JIT/harness 前检查 `flash_attn_interface.py` 与 `_vllm_fa3_C*.so`；缺失即 `rc=66`。因此当前处方是运行 `scripts/setup_flash_attention.sh`，或把该 env 指向已准备 checkout；不再要求删除目录再造 symlink。
  2. **bench gate fail=console 零输出静默 return 2**（stdout/stderr 双零字节）：死因只落在产物 json 的 `stderr_tail`/`diagnostic_stderr_tail` 字段——**"零输出 rc=2"分诊第一动作=读 json stderr_tail**（"假三件套先查 stderr_tail"纪律的 one-shot 版）。
  3. `--output` 为必选参数：缺它=argparse usage+rc 2（有输出形态），与坑 2 的零输出形态区分。

## §0.6 ★★★最终合仓修订（07-12 夜，晚于 §0.5，以本节为最终准绳）

- release `cuda-kernel` 已落生产提交 **`a8456f5`**：在 `3325c0d` 已包含的
  多轮 FA3 kernel 优化之上，同步 P1-B/L2 host/selector 终形态及审计根修。
  本增量没有改 FA gitlink/kernel patch，所以已经编过 `3325c0d` 的机器无需
  **再次**编 FA3；旧于该基座或全新安装仍必须正常编 FA3。selector C++ 实变，
  `selector_pipeline_ext` 必须按语义版本 `2026071201` 重建。
- §0.5 的“判据面零翻锚”结论已被严格单变量审计推翻：旧代码把合法
  `last_decode_refresh_step=0` 当成 `-1`，绕过 min-gap，使首个 producer work
  错误提前到 step 4。修复后为 step 16，work 序列从 `[4,99,100]` 恢复为
  `[16,111,112]`。仅回退这一个零值修复即可逐位恢复旧黄金锚，因果闭合。
- 用户已批准最终新锚转正：黄金 `{ad585b37,44e80946}`；bs8x12k 三发
  8-hash 均为
  `{fff0445e,c0a6376d,99a9aea2,7db6f96d,fda56972,deef29a8,8e8e2dca,8778b158}`；
  counts `{sentence:19,interval:16}`、payloads `756`；本地 A100 all-decode TPS
  `654.5–656.9`。旧锚只保留为 step-0 bug 的单变量复现证据，不再用于 release gate。
- Lite v1.1 与 CHUNK18 均未合入。SM80/SM90 合仓完成；FA4/SM100 patch 叠加链
  仍是独立遗留，禁止宣称 SM100 release-ready。

## §0.7 ★★★★合仓后本地优化续账（07-12 夜，晚于 §0.6）

> 本节描述 `a8456f5` 之上的**本地候选工作树**；尚未产生新的 release commit。
> 完整数据与接续命令见 `SFI_LOCAL_OPTIMIZATION_CLOSEOUT_2026-07-12.md`。

- L2 的 ATen post-topk 组合已换成定制 CUDA 稳定压紧：TP8 代理总耗时
  11.038→6.071 ms（1.818x），post-topk 7.239→1.847 ms（3.92x），峰值临时
  显存 766.31→37.45 MiB（−95.1%），checksum 不变。selector 语义升为
  `2026071203`，扩展/缓存名为 `selector_pipeline_ext_v2026071203`。
- 同名 `.so` 原地重编无法让当前进程的 dynamic loader 切换已加载映像；这是旧
  stale-dlopen 漏洞的根因。语义版本进入 module/cache name 后，旧 1201/1202
  即使仍在磁盘也不会与 1203 混载。远端只需一次 selector-only rebuild；本轮
  没有改 FA3 源码/gitlink/patch，**不重编 FA3**。
- runner 现在保留 child rc，并对 selector prewarm/postflight 的 path、sha256、
  actual/expected semantic 及 `run_provenance` 做 fail-closed 对账；每个 TAG 由
  非阻塞 `flock` 独占。VERDICT_ONLY 可接受的 sparse no-reference 噪声只有精确
  两项：`semantic_output_health_not_ok:unknown_without_reference`、
  `interval_trigger_intents_below_expected`；`async_producer_writer_missing` 是真红。
  default18 终复验已闭环：provenance 从 constants 单真源读到 18；VO route
  writer-complete count=36，与 full refresh-profile writer count=36 精确一致，
  producer reasons 为空、`SPEED RUN OK`。该 route count 是 measurement，不是
  allowlist；`async_producer_writer_missing` 继续 fail closed。
- capture scratch full 4B 实测 `[2,256,32,2,16384]` fp16=1 GiB：E2E child
  漏传 `max_num_seqs`，沿用 vLLM 默认 256，是真实过配。命令链 pin
  `max_num_seqs=batch_size=8` 的代码根修已落（针对性 12 passed），目标
  `[2,8,32,2,16384]`=32 MiB（32x/−96.875%）。根修后 full 4B c14/c18 的
  prebuild/live 均实测 32 MiB 且 live cache-hit，回收闭环。PBD 现实可回收
  0.408%，绝对上界 0.736%，no-go。
- CHUNK18 的 4B 六腿均值 658.0226→664.7054 TPS（+1.0156%），但 counts/payload
  从 19/16、756 变为 16/16、648，且 4/8 hash 改变。golden c14/c18 均
  gate/prod=True 且保持 `{ad585b37,44e80946}`/2+2/84；full 两形态也命中各自
  正典锚。因此本地 1203 候选已正式把**全局默认升为 18**，14 只作 env 回滚。

## §0.8 ★★★★★2026-07-13 本地终态（覆盖 §0.7 的 1203 中间态）

- 第一阶段“合仓位”已完成：完整源树与 `/ssd/xyxie/sfi-release-cuda` 的 29 个
  生产文件逐字节一致，包含 runner 的四个实际 benchmark 依赖和 postflight
  checker，不再存在“主树能跑、裁剪 release 缺 import”的隐式依赖。
- selector 终态 semantic=`2026071301`。L2 直接输出 canonical int32 pack order，
  L1 不再二次排序；旧 1203+L1 `1.977344 ms`，1301 `1.901568 ms`，10/10
  逐位相同。发布树 prebuilt SHA256=
  `995421642e5ff51ab2cf59d66b96a6f54c595d10dda162c658043387c9de694e`。这是
  selector C++/CUDA 实变，远端必须 selector-only rebuild；FA3 源/gitlink/patch
  没变，已有正确 FA3 基座不重编。
- host 热路径同时收口：prefill 36 层 lookup/bind/lease `36/36/36→2/2/2`；
  forward absent-phase mapping 分配 `36→1`，强 truth 路径少 216 个 device
  assert kernel；StepBound authority 缺失不 memo。
- infra 现在把 ABI cache、GPU/TAG 独占、KV 容量、FA3 checkout/build、精确
  corpus v2、nonce/artifact 和 8×256 token 完成度全部纳入 verdict-grade
  postflight。shared GPU 与 undersized KV 只可诊断，不能伪造正式绿车。
- A100/Qwen3-4B bs8×12k 最终配对：sparse `268.9904/665.8281`，dense
  `175.9740/292.7292`（decode/all-decode），即 `1.5286x/2.2746x`。最终统一
  gate `617 passed, 3 skipped`，红队 29 项 clean；29 个生产文件镜像一致。
- 仍未执行 commit/push；下一步只有取得用户危险操作确认后，才能在 release
  `cuda-kernel` 创建提交并推送。Lite v1.1 继续停靠，不混入此提交。

## §1 资产地图（写作时点实查）

### 1.1 待合三支（`git branch --merged main` 均为未合；merge-base 均=c56ed44）

| 分支 | tip | 文件面（diff --stat vs c56ed44） | 性质 |
|---|---|---|---|
| p1b-gap-throttle | e077c70 | patches/vllm_sparse_patch.py(+144)、utils/sentence_triggers.py(+12)、tests/test_trigger_interplay_contract.py(+95) | 换锚件（触发密度改变→counts/hash 必变） |
| l2-topk-tie | 27cdb87 | utils/selector_pipeline_ext.py(+60)、tests/test_selector_pack_order_determinism.py(+228) | 换锚件+**C++ ext 实变**（ext 重编+发布仓义务） |
| lite-v11-arm-body | cb9178e | patches/decode_runtime/metadata_builder.py(**+1373/−…**)、patches/decode_runtime/thin_builder_state.py(+49)、tests/test_lite_sig_return_contract.py(+627 新文件)、tests/test_thin_builder_state.py(+63)、tests/test_decode_dynamic_pack_gate_contract.py(±18)、tests/test_sm80_decode_lifecycle_contract.py(±12) | **拍板件**：判据全绿但结构性定谳=收益<噪声（命中 6/跑；38 长簇=设计内诚实 miss），1500 行大件零收益，推荐停靠保留 |

### 1.2 残骸/已消化分支（勿再合；甄别命令附后）

- 已消化（tip==merge-base，`--merged main`=1）：`chunk18-ab`、`pack-cuda-p1`、`small-sweep`、`u14-zero-alloc`、`ext-delta-rebase`。
- **`task16-zombie-batch` @2863ea5=残骸**：基座 876ffa1（很老），其主体（persistent_batch 整删等）已由 07-12 凌晨合入批以另一路径进 main——验证：`ls AXP/patches/refresh_runtime/persistent_batch.py` 应"不存在"。勿合此分支（会复活旧树形态），留档或删。
- **`anchor-window` @e8f41ae=活跃换锚窗验证树（=main+P1-B+L2 预合），勿删勿动**——它就是 §3 流程的载体，worktree 在 `.claude/worktrees/anchor-window`；拍板后按 §13.2 处置表收编。
- `lite-v11-pinned-guard` @e3a281a=已收编（内容进 main 9c561c1 R1'），可删；`ext-rebuild-batch`/`ext-delta-rebase`=真增量已由 7458eef 终局合入，留证据链阅后可删。
- 其余历史分支（arm-forensics-window/forensics-a2s1/j1-ko-speedgate-samesplit/refresh-amnesty/s1-ka-*/s1-kc-*/trigger-interplay-contracts/a2-fix-*）：均为已收针窗的工作分支。甄别通法：`git merge-base main <b>`==`git rev-parse <b>` ⇒ 已消化；否则先 `git log main..<b> --oneline` 看残余 commit 是否已有等价物在 main（grep commit message 关键词），不确定=留档不动。

### 1.3 成套铁律与判据锚（当前正典，换锚窗会更新其中判速部分）

- **FA vendored 成套**：FA commit `4c405dc` + `.so` 指纹 `fc8dbbcf8eaa`（122 elf sm_80）。多轮 kernel 优化已在 release 基座 `3325c0d` 中要求重编；`3325c0d→a8456f5` 不碰 FA，已编该基座者无需二次重编。若未来某支动 FA 面：换 `.so`=全机公共事件，须按 §6.3 先在交接档声明窗口。
- **本地 1203/default18 判速锚**：
  `{970c061e, c0a6376d, 99a9aea2, 03eb3325, fda56972, deef29a8, f6d008c1, 4076e3d5}`
- **release `a8456f5`/显式 chunk14 回滚锚**：
  `{fff0445e, c0a6376d, 99a9aea2, 7db6f96d, fda56972, deef29a8, 8e8e2dca, 8778b158}`
- **黄金 bs2 锚**：`{ad585b37, 44e80946}`（step-0 min-gap 根修后用户拍板转正）
- **counts 形态**：default18=`{sentence:16, interval:16}`/payloads `648`；
  rollback14=`{sentence:19, interval:16}`/payloads `756`
- **tps 参考**：同一根修后 full 单对照 default18=670.0263、rollback14=662.2989；
  远端绝对值以机器实测为准
- **hash 口径**：对 outputs json 每请求 `sha256(str(token_ids)).hexdigest()[:8]`；黄金档若没带 `--outputs-include-text`，outputs 的值直接就是 token_ids 列表。

## §2 判据发车配方（正典，含全部已知坑）

### 2.1 黄金 bs2（正确性硬门；qwen3-0.6b）
```bash
cd AXP && setsid nohup /ssd/xyxie/miniconda3/envs/vllm019-cu126/bin/python \
  benchmarks/bench_sm80_mixed_page_one_shot_graph_e2e.py \
  --output out/<tag>.json --summary-output out/<tag>_summary.json \
  --mode sparse --preset bs2long-cap128 --producer-mode full-open-gt1 \
  --model /ssd/xyxie/sfi/qwen3-0.6b \
  --python /ssd/xyxie/miniconda3/envs/vllm019-cu126/bin/python \
  --cuda-visible-devices <N> --skip-dense-reference --outputs-include-text \
  > <log> 2>&1 &
```
坑：①`--producer-mode full-open-gt1` 不可省（默认 legacy→child 踩 stale-runner 守卫 SystemExit 2，症状=route:missing 假红）；②`--outputs-include-text` 不带则 semantic gate 报 missing_text（hash 判据不受影响但 gate=False）；③绝不可直跑 run_sparse_only.py（裸 env 必踩"FlashAttention version not detected"假征状）。判读：json 的 `gate_passed`+`production_gate_passed` 双 True、`route=ResolvedRowPtr`、outputs hash 对 §1.3 黄金锚、counts={sentence:2,interval:2}/payloads=84。

### 2.2 判速档 bs8x12k（4B；换锚腿本体）
```bash
cd AXP && setsid nohup env VERDICT_ONLY=1 \
  PYTHON=/ssd/xyxie/miniconda3/envs/vllm019-cu126/bin/python \
  bash scripts/run_speed.sh <GPU> /ssd/xyxie/sfi/Qwen3-4B-Instruct-2507 bs8x12k sparse <tag> \
  > <log> 2>&1 &
```
PYTHON env 必带（fail-fast 设计，裸 python 毒化共享 ext 缓存）。
VERDICT_ONLY=1（vo）单发时间减半，判读键从 speed child 取。**§0.7 本地候选已
推翻旧口径**：`async_producer_writer_missing` 是真红，不能因 vo 放行；精确
allowlist 仅为 `semantic_output_health_not_ok:unknown_without_reference` 与
`interval_trigger_intents_below_expected`。`run_speed.sh` 会调用 fail-closed
postflight；产物=AXP/out/<tag>.json（+`_outputs.json` 为
dict{req_id:{text,token_ids}}）。必须使用唯一 TAG，并核对 selector prewarm/
postflight path、sha256、semantic=2026071203 全同。首轮换锚窗仍建议一发不带
VERDICT_ONLY 的全形态对照。

### 2.3 kernel-only 影子对（只有动 kernel/resolver 面才需要）
旧 `--case-filter non_tma` 默认 broad 可能让 production RRP 对全部 skip，形成证据
空集的 vacuous green；aggregate 已根修为无 RRP 证据即 fail closed。本轮 selector
负控的 pinned production 配方是。多 cell 聚合也已按
`(batch,requested_kv_len,case)` 隔离 missing/duplicate/launch/speed evidence，逐
record 检查 RRP，防止某 cell matched 掩盖另一 cell skipped（合同 25 passed）：

```bash
/ssd/xyxie/miniconda3/envs/vllm019-cu126/bin/python \
  AXP/benchmarks/bench_mixed_page_resolver_kernel_only.py \
  --case-filter kind0_native_non_tma_page64,kind4_rowptr_batch_row_non_tma,kind4_rowptr_irregular_stride1_non_tma \
  --force-num-splits 1 --output <json> --fail-on-gate
```

实测 `gate_passed=true`、max production RRP overhead=`0.215745%≤0.4%`。另有 broad
split1 三次 direct_physical=`0.428263%` 而红；该腿不是 production RRP，必须
诚实留档，但不阻断 selector merge。已知判读纪律：
`kind4_rowptr_irregular_non_tma` 的高税是构造性 stride-3 页集混杂，结构税以
`kind4_rowptr_irregular_stride1_non_tma`（iso-locality 控制腿）为准。

### 2.4 CPU 面（每支合入后顺手）
`pytest tests/ -q --collect-only | tail -2` 对账 collect 数（写作时=3102+lite 未合部分；每支合入应恰增其新测试数）；ring_war/interplay/pack 合同等定向文件按 §4 各卡片。

### 2.5 发车纪律（历史血账浓缩)
①每条 Bash 显式 cd 或绝对路径（cwd 会重置回主树根）；②发车前 `nvidia-smi --query-gpu=index,memory.used` 逐卡预检+`ps` 实查对端占卡（卡位表会过时）；③预检、发车、确认存活**分三条命令**（`预检 && nohup ... &` 会把整链后台化吞输出）；④pgrep/pkill/Monitor 模式必须 [x] 转义（否则匹配监视器自身=死监视）；⑤后台长跑用 setsid nohup+落盘日志（Bash 工具 10min 上限）；⑥判速判读先查 `run_provenance.git_head` 防判错树。

## §3 合流总流程（换锚窗六步）

**第 0 步（拍板，阻塞项；历史流程）**：向用户呈 §8 拍板件，至少要拿到：①Lite 合/停靠（推荐停靠）；②CHUNK18 生产默认 `_CAPTURE_CHUNK` 14→18 改不改。**后写的 §0.7 已完成第②项并 promotion 为默认18，接续者勿重拍/重跑。** P1-B+L2 已有既定方向（记录在案的"下一换锚窗=P1-B+L2"），无需重拍。

**第 1 步（定稳定基座）**：`git log --oneline -5` 确认 main HEAD 且两场均无在飞 commit（§6.1 交接档尾查）；在交接档写"换锚窗开启+排他声明"（对称礼仪：既往 host 场合入窗同款）。**写作时点此步与第 2、3 步已由 host 场以 `anchor-window` 预合树推进中（见 §0）——接续者从 §13.3 回填状态判断切入点，勿盲目从头跑。**

**第 2 步（逐支 rebase+判据，按依赖序）**：
- 顺序建议 **P1-B → L2**（→Lite 若拍板合入）。理由：P1-B 冲突面最小先趟平基础设施；L2 有 C++ ext 重编义务放后集中处理；Lite 面最大且默认不合。
- 每支：`git rebase main <branch>`（或 worktree 内 rebase，见 §6.4）→解冲突（逐支冲突预案见 §4/§5）→**先跑该支自带判据**（§4 卡片"判据义务"）→绿则 `git merge --ff-only`（或直接把 rebase 后分支快进给 main）→CPU collect 对账。
- **组合形态原则**：每合入一支后判速 vo 一发看 counts/hash 形态变化并记录（此时 hash 允许变=换锚件本性；黄金锚必须仍逐位，黄金若变=立即停手回退定位，黄金不在本轮换锚域）。

**第 3 步（组合总闸）**：全部合完跑：黄金（2.1，gate 双 True+锚逐位）+判速 vo（2.2）+定向 CPU/GPU 测试面（§4 各卡片并集）+collect 对账。L2 在列时 kernel-only（2.3）也跑一发（它动 selector 而非 attention kernel，预期零影响=用作负控）。

**第 4 步（换锚腿 ×3 自证）**：同卡同配方判速 3 发（间隔可 10min 级），要求 run 间 8-hash **逐位互同**（topk 决定论修复后的既有承诺域=bs8x12k 锚工况 run 间 bitwise）+counts 逐发相同。3 发同 ⇒ 产出候选新锚组（8 hash 值+新 counts+payloads+tps 新参考带）。**任何一发互异=不许换锚**，回到逐支二分。

**第 5 步（呈拍板+落档）**：候选新锚组呈用户拍板转正。拍板后：更新交接档头注锚行+RUNBOOK 判读五关（`SFI_KERNEL_SPEED_GATE_RUNBOOK_*.md`）+REMOTE_TEST_INSTRUCTIONS（新锚+"重测以逐位复现为判据"话术沿既有模板）。**旧锚保留一行"单变量自证工具"注记**（pin 回旧形态可逐位复现旧锚=换锚合法性证明，先例两次）。

**第 6 步（发布仓同步+push）**：见 §7。

## §4 逐支 merge 卡片

### 4.1 p1b-gap-throttle（P1-B，[GAP-THROTTLE-GENERATIONAL] 方向 B）
- **内容**：世代 gap 班车节流直译+下游三传票复审必要项；sentence/lease 形班车保持全额 gap（避 A1 同型崩链的收窄决策）。
- **判据义务**（分支 commit 正文有欠账单原文）：interplay 合同 19/19（含 T8 新语义+T8b/T8c）；周边双态逐位同；CPU 复现 gap48 班车 nreq 3→7=方向 B 预期值；合入后判速 counts 形态记录（触发密度变 ⇒ sentence/interval 计数会漂，属换锚预期）。
- **冲突面**：vllm_sparse_patch.py/sentence_triggers.py——main 侧 c56ed44 后无人动（写作时实查），预期 rebase 干净；test_trigger_interplay_contract.py 同。若实操冲突：触发域语义以分支侧为新、以 main 侧既有 AMNESTY（信号合一）语义为底，两者都保。
- **特殊义务**：无 .so/ext 义务（纯 python patch 层）。**用户既往重申：任何 refresh 触发路径改动必须复核 sentence×interval 两触发交互面**（TP8 bug 根源史）——interplay 合同全绿即此义务的载体。

### 4.2 l2-topk-tie（L2，selector topk 并列决定论两段法）
- **内容**：边界并列带二段修复落 selector_pipeline_ext.py 两臂咽喉（C++）；消 topk 并列的 run 间非确定残余。
- **判据义务**：分支自带 12/12 单测+C++ standalone GPU 30/30 逐位==oracle；
  §0.7 已完成定制 CUDA 深化，相关 selector 套件 87 passed、整合定向套件
  89 passed，覆盖 graph/±inf/NaN fail-loud。旧 ATen 组合税已不再是遗留；
  checksum 同且新总耗时/raw top-k=1.496x。最终跨 runner/capture/kernel/
  producer/selector/one-shot/arena 合同并集 **436 passed**。default18 的 262K
  arena 预算已从 chunk/in-flight 单真源推导为 5 GiB，live i32 seq-lens 也在
  预备阶段落位，避免 refresh cache-hit 补建。
- **冲突面（已被 host 场预合实证降级）**：`selector_pipeline_ext.py` 在 main 侧 c56ed44 之后被改三轮=`01bd4fe`（F1 shim）→`e234a33`（p2 守卫+p4 env 严格化 hunks）→`2db71ca`（刀A ensure 三层去重）。**host 场 §13.2 定谳：L2 的 +60 行与刀A/刀E 同文件不同区=自动合安全（`anchor-window` @e8f41ae 预合树实证）**——正常路径直接用该预合树/自动 merge。仅当需要**重新** rebase（如 main 又前进且自动合冲突）才用手工预案：逐 hunk 对齐、L2 并列修复+main 侧守卫/去重全保留；解完跑 C++ standalone 30/30+p4 env 严格化负测（`VLLM_SPARSE_SELECTOR_FUSE_NMS_CROSS=01` 起一发应 raise）双向确认。
- **特殊义务**：C++ 实变 ⇒ ①selector pipeline 首跑 JIT 重编（数分钟；当前
  semantic/module=`2026071203`/`selector_pipeline_ext_v2026071203`）；②prewarm
  与 postflight artifact hash 必须不变；③**发布仓义务**：
  selector_pipeline_ext.py 与 runner/provenance 生产文件随 §7 同步（FA `.so`
  不涉）。

### 4.3 lite-v11-arm-body（Lite v1.1，默认不合——先看 §8 拍板）
- **内容**：SIG_RETURN 返程步增量臂完整形态（含 rebind；四 commit 链 6fae919→f38860c→a5c8740→cb9178e，截存缺口根修+消费点终位）。
- **既有定谳（合入决策的事实基础）**：判据全绿（8-hash 8/8×8 发+counts×8+黄金×5+CPU 双态失败集逐条同）**但结构性收益=命中 6/跑+38 长簇诚实 miss（页界/recent 窗相位=必走 PBD 慢路径，extrapolate 拒=正确行为）；tps 七发 665.9-670.1 与合入前无可辨差**=+0.8-1.2% 预期在现 HEAD 不成立（慢路径已被 K-a/K-c/L1 多刀瘦身，臂边际收益缩水至噪声下）。
- **若拍板合入**：冲突面大——metadata_builder.py（+1373）与 main 侧 `6d02463`（u14 #14：_compact_staging numpy 视图化等）同文件双方大改；thin_builder_state 与 main 侧近期改动亦有交叠。rebase=重活，预算半天+全量判据。**若拍板停靠（推荐）**：分支保留即可，其 J3 仪器五键+refuse trace 已随 main（下窗判读资产），无进一步动作。
- **衍生拍板**：PBD 形态臂立项与否（38 诚实 miss 转命中=剩余收益面；extrapolate 放行页滑+plan 整建重铺 first/count；新刀独立判据另走，不在本合流窗）。

## §5 冲突面总表与 rebase 实操

### 5.0 动手前重新实查（main 可能已前进）
```bash
cd 主树
git log --oneline -8                       # 双场是否有新 commit
tail -30 SFI_NIGHT_SESSION_HANDOFF_2026-07-11.md   # 对方有无新声明（§6.1）
for b in p1b-gap-throttle l2-topk-tie lite-v11-arm-body; do
  echo "== $b"; git diff --stat $(git merge-base main $b)..$b | tail -3
  git log --oneline $(git merge-base main $b)..main -- \
    $(git diff --name-only $(git merge-base main $b)..$b) | head -8   # main 侧同文件后续改动=真实冲突面
done
```

### 5.1 已知交叠（写作时点）
| 分支文件 | main 侧 c56ed44 后改动链 | 风险 | 预案 |
|---|---|---|---|
| selector_pipeline_ext.py (L2) | 01bd4fe→e234a33(p2/p4)→2db71ca(刀A) | 高 | §4.2 解法；两侧语义全保留；解后双向判据 |
| metadata_builder.py (Lite) | 6d02463(u14 三刀) | 高（仅当拍板合入） | 双方段落基本不同区（u14=descriptor 写段；Lite=SIG_RETURN 臂段），冲突多为上下文行；以"两者都保"解 |
| sentence_triggers/vllm_sparse_patch (P1-B) | 无 | 低 | 直 rebase |
| tests/* 各支新测试 | ccbf85e 动过 rrp/mixed_page/key_norms 三文件（与三支零交集） | 低 | — |

### 5.2 前车之鉴（同类冲突已发生过的案例，供判断风格）
- `test_req_meta_pack_sink_validation.py`：kernel 场（a703cc9 锚修+迁测）与 host 场（3823c50 顺手根修）同文件双改——最终两边语义并集共存，无回退。
- host 场 ext 增量四件中两件（A1 案B/shim 7→8 列修）在 rebase 时发现**宿主已被对端删除**（alpha_selector_kernel 整删/7 列桥整删）→正确处置=**弃件**（正确性目的已由删除达成），见 `7458eef` commit 正文。教训：rebase 前先确认自己 hunk 的宿主文件/函数还在。

### 5.3 rebase 操作形态
- 在独立 worktree 里做（`git worktree add ../wt-<branch> <branch>`），不动主工作树；rebase 完成+判据绿后再快进 main。
- **并行窗禁用 git stash**（仓库级共享栈有竞态+曾丢 tracked 改动的血案）：需要双态对照时用"先 commit 再 diff/checkout 对照"。
- 每支合入 main 后立即在交接档滚动账记一行（commit hash+判据摘要），供另一场实时看见。

## §6 多 session/多 agent 协作纪律（血账浓缩，必守）

1. **commit/push main 前必 `tail -30` 交接档**查对方新声明（排他窗声明可能写于你最后一次读档之后——07-12 已实际发生一次时间线交叉）。撞窗处置=立即写应答条目（含给对方的冲突面清单）+冻结 main 至对方窗关。
2. 开工前 `git branch -a`+读交接档最新滚动账——**对方可能已把你清单上的活做了**（07-12 两例：#14、CHUNK18 均撞车，处置=skip+独立核验互证）。
3. **换 FA .so/重编公共 ext 前在交接档声明窗口**（乒乓竞态曾致判据车全红的"一次性环境事故"）。
4. worktree 三件套：独立 worktree+独立 TORCH_EXTENSIONS_DIR+发车前 ps 实查占卡。
5. **ext 乒乓**：TORCH_EXTENSIONS_DIR 默认=~/.cache/torch_extensions 按 ext 名字键缓存；两棵源树轮流用同一缓存=每次全量重编+FileBaton 互等。多 session 期用私有 dir（一次性预热 ~15min）。黄金 one-shot 驱动自带 AXP/tmp/torch_extensions/sm80_gt1（仓内私有）。selector 从 semantic 1203 起把版本写入 module/cache name，并统一禁用 NVCC depfile 生成，已消除同名 stale dlopen 与 import 顺序导致的重复 JIT；跨语义升级仍应使用新私有 dir。
6. 判读必查 `run_provenance.git_head`；跨窗 hash 对比先确认口径（sha256(str(token_ids))[:8]）与形态（counts 类集合）。

## §7 发布仓同步与 push

- 位置：/ssd/xyxie/sfi-release-cuda（分支 cuda-kernel）。**只 push 这个仓**（外层老仓不 push；主树无 remote）。push 本身需用户授权（历史惯例：授权可以是对一批的概括性授权，如"授权处理"）。
- **裁剪三类不同步**：debug/test 件（发布仓无 tests/）、判据工具（一般为 check_*/profile_*/bench_mixed_page_resolver_kernel_only 等）、oracle 测试件（attn_reference.py 历来不带）。**例外**：`scripts/check_run_speed_summary.py` 是 `scripts/run_speed.sh` 的 fail-closed 生产 postflight，两者必须作为原子生产单元同步。
- 同步法（源码级）：逐文件 `diff -q` 主树 AXP vs 发布仓同路径（utils/patches/triton_kernel/hybrid_selectors/benchmarks/scripts 六目录）→DIFF/NEW 甄别（新文件先判死活与裁剪类）→cp→发布仓 commit（英文 message，模板见 `3325c0d`/`fc5b0d1`：写清行为合同、ext 重编义务、FA .so 是否受影响、锚是否不变）→push。
- **FA .so 义务判定**：只有 FA vendored 树（gitlink）改动才触发 kernel_patches/sfi_fa3_sm80_sm90.patch 重生成（`git diff --binary f5bc33c..<新 FA commit>`）+远端"必须重编 .so"指引。本轮三支均不涉。
- 远端指引=主树根 REMOTE_TEST_INSTRUCTIONS_2026-07-12.md（追加"增补"段的先例格式在档内）；换锚后必须更新锚值段。
- 已知遗留：fa4/SM100 patch 与新 fa3 patch 叠加链断裂（h1/h2 上下文漂移），SM100 用户暂用上一版组合——**下一个动 FA 的窗顺手 rebase**（README 已注记）。

## §8 拍板件清单（合流窗开工前呈用户）

| # | 拍板件 | 数据面 | 默认建议 |
|---|---|---|---|
| 1 | Lite v1.1 分支处置 | 判据全绿但收益<噪声（tps 七发无可辨差）；1500 行大件 | **停靠保留**（墙钟纪律） |
| 2 | PBD 形态臂立项 | 现实可回收 0.408%，绝对上界 0.736% | **no-go**（低于 1% gate，不投 1500 行） |
| 3 | CHUNK18 默认 14→18 | 六腿 +1.0156%；golden/full 两形态各自锚闭环 | **已 promotion 为默认18**；14 保留 env 回滚 |
| 4 | L2 深化（定制 CUDA） | 11.038→6.071 ms；临时显存 −95.1% | **整车组合复验已完成，release 工作树已本地同步**；待 commit/push 明确确认 |
| 5 | selected_k 降档/P-1/P-3/LongBench | 历史搁置件 | 维持"精度不测"拍板，不进本窗 |

## §9 本窗之后的其它遗留（非合流，登记备查）

- oracle 生态终态：triton 全下线完成（a703cc9）；数值 oracle=①同 ext scalar 臂（reduce 对拍）②tests/reference_log_f_prior.py（fused 族 torch 参考，以 C++ 为正典；**其中记录了一处 C++/triton 历史数学分歧**=prior one_minus 单 token 窗，已按 C++ 裁决）③attn_reference.py（活 oracle）④pack 族 triton kernel 留作 req_meta_pack_ext 的逐位对拍 oracle。
- 远端反馈待收：320e5fb+fc5b0d1+6e2763d+3325c0d（TP8 机器重测）。
- 20+1 预存红已全数追平（ccbf85e），该债务已清；若三支 rebase 后这三个文件再红，先查是不是分支侧旧期望复活（以 main 侧 ccbf85e 的追平语义为准）。
