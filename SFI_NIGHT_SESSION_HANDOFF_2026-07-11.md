# 07-11 深夜场交接（主入口；★所有推进中任务均未完成，含一个跨 session 阻塞）

> 本档为新主入口，接替 `SFI_HOTPATH_REMAINS_HANDOFF_2026-07-11.md`（其 §1 地图/§2 清单/§4 工具/§5 判定书仍有效；§-1 收案补记被本档 §5 索引取代）。发车配方/坑库=CAMPAIGN §3/§4 不变。
> 判据链=黄金锚 {0d5f663c,636fb032}+TP1 判速带（595-607）+8-hash 锚组+counts（口径=route.jsonl payload_plan reason 计数，基线 {sentence:30,interval:4}）+collect（当前 3016）。
> **★07-11 晚拍板/勘误更新**：黄金 bs2 锚已由用户拍板换新=**{636fb032, 44e80946}**（K1 终账 §3 选项 A；pin1 还原口径保留自证）；判速带现参 K1 终账 §6.1（≈605-612）；collect 真值=3000（§8.2 勘误）；counts 以"事件计数 19 类逐类全等"强口径替代 reason 细分（K1 终账 §6.1 诚实注记）。
> **★★最新判据基线（07-12 起以 §10 为准,上一行判速带已过时）**：判速 tps **671-673 带**+**8-hash 新锚（SPLIT-ROOT 拍板组,见 §10）**+黄金 {636fb032, 44e80946} 不变+counts 19 类口径不变+**.so 成套铁律 d5a692**（旧 .so 遇 split>1 启动崩）。**全量推进总序=`scratchpad/GPU_WINDOW_BATTLE_PLAN_2026-07-11.md`（已升 v2 总作战单:六轮序+窗 1 详单+推送批清单）**。
> **★★★07-12 上午拍板批（以本行为最新判据基线，上两行已过时）**：用户对呈报拍板件**全部授权**（新锚转正/P1-B 立项/gap 24→32/K-a·K-c 兑现账/发布仓推送/P-2 立项/oracle 同 s 化）。**8-hash 判速锚正典 = {242dfec6, b18ff00c, 58b3b21e, 2017b63c, f0b76039, cc46a168, d04130be, e57dc3a4}**（AMNESTY+topk-L1 叠加形态，run×3 逐位自证=跨 run 稳定已恢复）+黄金 bs2 **{636fb032, 44e80946} 不变**（换锚批后补验逐位，1838611）+**counts 新形态 refresh_reason_counts={sentence:15, interval:16}、payloads=612**（口径=类集合+hash 硬门、reason ±1 容差）+tps 正式带**待独占批重立**（参考：645-649 load 参考 / 686-687 kernel 场环境口径）+**.so 成套铁律=04cbbf9794fa**（FA b08420e=N1+L2R2+N3+N4-reopen；**07-12 P-2 后成套已更新=FA `4c405dc`+.so `fc8dbbcf8eaa`，tps 正式带已立=663-667，见 §12**）。拍板落档与推进账=**§12**。
> **★★★★07-12 傍晚合流入口**：停靠分支合流（kernel 侧+host 侧优化 merge）的**自包含手册=`SFI_MERGE_GUIDE_2026-07-12.md`**（资产地图/判据发车配方全量/逐支 merge 卡片/冲突预案/发布仓与多 session 协作纪律；**接续 agent 从该档 §0 进入**），与本档 **§13**（host 场收工终账+分支处置表+换锚窗 `anchor-window` 实跑进展）互补对读；细节出入以更晚写入者+实查为准。

## §0 一屏状态

- **main 至 `c55fe95`**（07-11 全场我方 ~19 笔+对端 kernel 场 2 笔混入；无 remote 本地留档）。
- **★首要阻塞：main HEAD 判据不可发车**——对端 kernel 专场 `3f39bf1`（16 刀：capture half2/probe managed 税/SM90 WAR/死码-565）04:17 落共享 main，其上两发 bs8x12k 判速=**229 tps（崩 62%）+8-hash 组Y MISMATCH+counts 漂移（sentence 32）**；对端自验仅黄金档 bs2，判速档从未跑。**解除条件=对端复核**（红则按其 K1-K7 刀单二分回退）。判读工具=产物 `run_provenance.git_head`（=88a8072 及以前的判据有效；含 3f39bf1 的作废）。
- 污染前最后绿基线：`arm_revert_probe`（head=88a8072）=600.2+8-hash MATCH+counts 逐位。
- 发布仓积压 ~14 笔待授权（**三 env 已转正=prebuilt .so 必须重打**）+对端两 .patch.new 待授权。远端 TP8 按 REMOTE_TEST_INSTRUCTIONS_2026-07-10 重测中。
- 工作树残留：FA gitlink 脏（对端 kernel 场进行中产物，勿动）；`scratchpad/lite_arm_body.diff`（1252 行臂体 v1 全量，返工母本）。

## §1 未完成任务全清单（诚实状态）

| # | 任务 | 状态 | 已完成部分 | 剩余 | 入口档 |
|---|---|---|---|---|---|
| 7 | **Lite P0 臂体** | **红案回退，v1.1 待返工** | 零件四批已 commit（枚举/快照/epoch 门/外推函数/遥测三键，判据绿）；臂体 v1 全施工+16 合同测试（在 diff 里随回退离树）；红案 6 发定谳（铁证=组X 定罪；洗清=admit/plan/delta 全链；segfault 归因降级） | v1.1 返工：先跳 RRP rebind 立正确性（跳=下一稳态步 q_layout_key_mismatch 跌慢路径，收益打折但可判）→绿后 rebind 单独接回+**GPU 页表逐位 dump 对拍**（heavy bind 后 vs 臂 rebind 后 row_table_i32/seqused） | 锚点手册 §13 施工序+**§14 及两补=红案档（返工必读）** |
| 8 | S1 sync 孤洞取证 | 未开工 | 真刀单定位（18ms cudaStreamSynchronize 三连，+0.5-1.7% 上界） | z 探针+调用栈归因，半天 | 真刀单 #2 |
| 9 | BUG-A2 抢占窗 | 未开工 | 审计定位（resume reset 不清票/时钟） | serve 抢占场取证→resume reset 清票+last=-1 | 审计档 §B |
| 10 | 广域挖掘 | ✅ 完成 | 六新立项+四报告入仓 | — | — |
| 11 | CHUNK18 AB | 未开工（**被 §0 阻塞**） | 可行性放行（全消费端参数化）+G4 雷已拆（A2 raise） | 同卡交替 AB+黄金（0.6b 变 18/10 同跑）+capture 计数对账 | 新方向 #1 |
| 12 | 判据链瘦身 | 未开工 | 结构定位（diag child 无条件跑=45s 浪费） | route_proof 数据源核查（30min）→--verdict-only+5min 先导判 | 新方向 #2 |
| 13 | 触发密度曲线 | 未开工（**精度轴=用户拍板件**） | 量化（+2.5-4.5% 上界，TP8 加倍） | 速度轴 {24,32,48} 同卡 AB 半天→LongBench 14 子集 1-2 天→呈拍板 | 新方向 #3 |
| 14 | ultra 零分配束刀 | 未开工（**推荐解除阻塞后首刀**） | 定位（launch_template 16 次 setitem=80ms/跑=U3 本体；2-1 单落仅 15 行） | numpy 视图化+判据（顺带 decode_p95 前后对比结案 U3） | 考古 #1 |
| 15 | per-layer 上提束刀 | 未开工 | 定位（36 层重算步级量 0.9-1.6ms/eager 步，a/b/d/e/f 五件） | 上提 _ensure_step_prologue+B 洞碎事件 1155 前后对比 | 考古 #2 |
| 16 | 僵尸批删 | 未开工（depth-3 fail-fast 已单独落=8a98e52） | 清单全定界（persistent_batch 413+双 ext 影子 610+page_sparse 15 字段+危险旋钮） | 批删+collect/判据 | 考古 #3 |
| 17 | triton 下线（用户指令） | 活性定谳完成，未动树 | alpha_selector_kernel=死 import 可整删（A1 长 ctx 雷随删消灭）/req_meta_flag_codec 可搬家/pack 族=生产主链 | P0=死件删除批（alpha_selector+attn_reference+alpha_fair 两死函数+mixin 死 import）；P1=pack 迁 CUDA（C++ 窗 128µs 实证可行） | #17 任务书 |
| — | ext 高危 #2（accum 列宽 CHECK≥10）+ext 卫生 14 条+姊妹漂移 | 未修 | 审计全录 | 并入刀A/D/E 的 ext 重编窗一次做 | EXT 审计档 |
| — | 发布仓同步 | 待授权 | — | ~14 笔+prebuilt 重打+对端 .patch.new | 记忆纪律条 |

## §2 作战序列（依赖排好）

1. **解除 §0 阻塞**：对端复核 3f39bf1 判速档（或用户授权我方代跑：在 3f39bf1^ 与 3f39bf1 各一发判速 AB 即可定罪其刀单；红则对端二分回退）。
2. **快刀恢复节奏**（阻塞解除后）：#14 的 2-1（15 行 80ms）→#12 判据瘦身（其后所有刀迭代 ×2）→#17 P0 死件批+#16 僵尸批删（零风险清淤合并一批）。
3. **#11 CHUNK18 AB**（半天，一刀三鸟）。
4. **臂体 v1.1 返工**（红案档配方；建议专场——首判据直接判速档+GPU 页表 dump 对拍工具先行）。
5. **#13 触发密度**速度轴→精度轴→呈拍板。#8 S1/#9 A2 穿插。
6. 发布仓授权同步（含 prebuilt 重打）。

## §3 本场新纪律（三条，已入记忆）

1. **共享树判据纪律**：判据发车前 `git log -1` 锚定；判读必查产物 `run_provenance.git_head`；多 session 并行窗内的判据一律标注树版本。
2. **大施工首判据=判速档**（黄金 0.6b 28 层踩不到 36 层/bs8 雷区——本场臂体红案被黄金绿骗了一轮）。
3. **深水"照抄改造"段（>100 行）必配逐位对拍工具再上判据**——合同测试只钉调用形态钉不住语义；GPU 侧写必须 GPU 侧 dump 对比。

## §4 本场定谳索引（勿重取证）

- **真刀单** `SFI_TRUE_KNIFE_LIST_2026-07-11.md`（6c2d567）：bs8 慢步=GPU 等 host 断流（bind 背压仅 bs2 档）；FLFP/bounded-topk/writer 带宽三刀封路；Lite 刀R +0.8-1.2%（判据=trace B 洞 9-13→4-6）。
- **新方向** `SFI_NEW_DIRECTIONS_2026-07-11.md`（d4edbbd）：CHUNK14=0.6b 遗产/判据瘦身/触发密度/prefill 段 gap 地图/显存首盘/rank0-broadcast 永久封路。
- **代码考古** `SFI_CODE_ARCHAEOLOGY_2026-07-11.md`（0f4991b）：U3 破案=launch_template setitem 80ms/per-layer 税清单/僵尸清单/假遥测 per_step_allocation_count。
- **EXT 域审计** `SFI_EXT_DOMAIN_AUDIT_2026-07-11.md`（bb2f67d+两补）：kLogFPreMaxR/accum 列宽/K>65536 LSE 三高危+阴性锚全录。
- **触发×short-dense 审计** `SFI_TRIGGER_SHORTDENSE_AUDIT_2026-07-11.md`（35276ee）：打架面四面安全+八条 fail-close+BUG-A1 修。
- **Lite 锚点手册** `SFI_DUAL_GEN_LITE_P0_ANCHORS_2026-07-11.md`：§1-§12 锚点+§13 施工序+§14 红案档。
- P3/B' 收案=`SFI_P3_PREWARM_DESIGN_2026-07-11.md` §7-§8（prewarm/defer 双否决，学费即地板）。
- 已落防御收窄：depth-3 (1,2)/REDUCE_GROUP 三口 raise/last_n>16 raise/B2 对齐/BUG-A1 crossing 豁免/inflight_compact_readable 代理/快照失效钩/flush 遥测三键（★DecodeRuntimeCounters 全族历史无遥测=旧计数判读全是假象）。

## §5 坑新增（并入 CAMPAIGN §4）

- **并行树污染**：另 session commit 切开判据链（本场 dbg3/final_green 作废）；ext JIT 缓存共享=对端 kernel 源改动使 4 ext 静默重编。识别=tps 崩+新错 hash 组+git_head 对不上。
- **segfault 边界归因**：施工窗内的崩溃不能直接归因给自己的改动（对端未 commit 工作树可能已在共享树上）。
- cwd 七犯（并行块/py 内 open 两新变体，纪律第 9 条）。
- 黄金锚绿≠判速安全（§3-2）。

---

# 交接补遗 v2（用户令：不返工，先弄清为什么红+未落地项全录）

## §6 臂体红案根因分析现状（截至本场收针，root cause 未定但嫌疑域已收窄）

### 6.1 阻塞解除更新
对端 half2 案已收案（8b3c2c1 回退+612.1 绿；root cause=SM80 nvcc 放弃展开→acc spill STACK 112→1232；本线 229/组Y 指控被采纳）——**main HEAD 判据解冻**。组Y 归因闭环=half2；**组X（臂体案）与组Y 是两个独立红案**，时间线互证：revert_probe（88a8072，两案之间）绿。

### 6.2 组X 案已证事实（铁证层）
- 臂命中 5 次/跑（step 111/159/207/378/426，世代 48 步节奏），错 hash 组X deterministic（J1 与 repro 逐位同）；**2/8 请求幸免（58b3b21e/cc46a168）**。
- **臂 host 值面 100% 正确**（dbg3 打点终证：real/first 同步推进、visible=数学周期同值、外推 vs collect 逐位吻合、assert 从未炸）。
- 错值面=GPU 消费侧（页表/req_meta/binding 之一被写错），行子集错（6/8）。

### 6.3 静态排除链（六嫌疑逐一判别，勿重查）
| # | 嫌疑 | 判别 | 结论 |
|---|---|---|---|
| 1 | repack dummy capture ptr=0 与慢路径不一致 | 读慢路径 :7274-7292——无 logf 行时同样传 dummy(ptr=0+全0 row) | **排除**（逐字节同款） |
| 2 | 触发步 mask=1 残留 GPU 缓冲（脏标未置→臂归零被跳过） | 触发步有-rows 臂 :6627-6632 无条件重算脏标（mask 有 1→True）→臂 `_ensure_decode_logf_no_rows_buffers` 归零必执行 | **排除** |
| 3 | bind 读触发步 StepDecodeData 构建 binding | bind :4848 只对 SDD 做 None 检查；q_lens/row_mode/context/epoch 全从 **step_authority**（本步真值） | **排除**（输入面对） |
| 4 | geometry 读旧 plan | bind :5126/:5138 从 `self.step_bound_meta.compact_recent_launch_plan`（臂步 7 已换新 plan）取 | **排除**（载体对） |
| 5 | q_layout_key 带 q=2 | key 从 authority q_lens 烘（:76 注释+读点 :50-63）=本步 q=1 | **排除**（且即使错也只慢化不改值） |
| 6 | WAR fence 缺失 | maybe_build 入口 wait（:6042）盖住 inner 内全部（臂在 inner 内） | **排除**（结构性被盖） |

### 6.4 剩余假设（静态不可判，按概率排序）
1. **manager dirty-diff 与臂时序的隐式耦合**：`manager.update()` 的七元组 diff 以 manager 存储的**触发步 layout** 为基准产出 dirty 行——但触发步 heavy bind 后 manager 的 `_row_table_layout_by_row` 是否真的已带 capture-full 形态（若触发步某路径没更新 layout 影子，diff 会漏 capture 行=行表不重铺=**读 full 页表但 descriptor 已是 compact 语义=错值**，且漏的行数=capture 行数——**与 6/8 错 2/8 对的形态可吻合**：若触发簇里 6 行是 capture/logf 行 2 行不是）。
2. **臂 bind 与臂内更早 GPU 写（repack/xlayer copy_）的同流序**：均在当前流理论有序，但 bind 内部若切流（publish staging 用专用流/事件），臂的 copy_ 与行表写的相对序与慢路径（bind 在 repack 之前！慢路径序=plan→SDD→**bind→bound_meta→xlayer→repack**）**相反**——臂序=bound_meta→xlayer→repack→bind。若 kernel 消费面对"req_meta 先于行表"有隐式假设，序反=窗口。
3. **`_record_resolved_row_ptr_owner_update_ready_event` 的 count 语义**：臂置 `bind_call_count=0` 再断言 ==1——该计数器若是臂 agent 新造的（bind 内自增点需核实），断言恒过但 **ready event 可能未按臂的真实 kernel 数记录**→重放流不等臂的 H2D=读到半写行表（**每发同一批行先到先错=deterministic**）。
4. 触发簇形态假设破缺：臂只验证了 logf 返程（q 恒 1）形态；若命中步实际是 q 返程（触发步 q=2）而 repack 的 q_lens 缓冲复位语义有别——但 hits 里 q 面已翻回（admit 保证），概率低。

### 6.5 下场取证配方（非返工，纯定位；预算 ~2 GPU 轮+半天）
1. **GPU 页表逐位 dump**（决定性）：臂树+debug env 下，在臂步 12 bind 完成后 dump `row_table_i32[:bs*heads]`+`arena_batch_seqused` 到文件；同一步强制 fail-close（临时开关）走慢路径全量 bind 再 dump——逐位 diff。错行集合直接显形，与"2/8 幸免"的 req→行映射对上（outputs 的 req_id 序=行序）。
2. manager layout 影子核查（假设 1）：触发步 heavy bind 后 print `manager._row_table_layout_by_row`——capture 行的七元组是否 full 形态（valid=0）。若不是=diff 基准坏=假设 1 实锤。
3. ready event count 审计（假设 3）：grep bind 内 `_decode_runtime_rrp_metadata_bind_call_count` 自增点；核 `should_record_rrp_ready_event` 在臂调用链的真实走向。
4. 行为对照弹药已备：`scratchpad/lite_arm_body.diff`（v1 全量）+hits log 配方（`VLLM_SPARSE_LITE_SIG_RETURN_DEBUG_LOG`）+污染前绿基线 head=88a8072。

## §7 全部未落地开发/优化项与 findings 总录（按来源报告；交接前状态=全部未动树）

### A. 真刀单（SFI_TRUE_KNIFE_LIST）未落地项
- Lite 刀R（**红案回炉**，+0.8-1.2%）；S1 sync 孤洞取证（18ms×2-3，+0.5-1.7% 上界）；刀A/D/E+writer enqueue 凑批（+0.3-0.5%，ext 重编窗）；Full 刀（score-only，+2-3%，**换锚拍板件**）；selected_k 降档（+5.4% 实证+容量红利，**精度拍板件**）。
- 封路结论（勿再走）：FLFP/bounded-topk/writer 带宽/错峰摊派/prewarm/defer-capture/rank0-broadcast。

### B. 新方向报告（SFI_NEW_DIRECTIONS）未落地项
- CHUNK18 两均分（一刀三鸟，半天 AB；G4 雷已拆）；判据链瘦身（verdict-only+5min 先导判，迭代 15→5-7min；前置=route_proof 数据源核查 30min）；触发密度曲线（min_gap 速度轴半天+LongBench 精度轴 1-2 天，**+2.5-4.5% 上界全场最大，拍板件**）；bootstrap 兑现窗挪 prefill 伞（e2e ~60-80ms，前置=nvtx 打标定谳 SFI 专属性）；topk pool 显存 snapshot（半小时消 UNKNOWN）；slot_by_row 跨 rank 合同校验一行。

### C. 代码考古（SFI_CODE_ARCHAEOLOGY）未落地项
- **ultra 零分配束刀 2-1**（launch_template numpy 视图化 15 行=80ms/跑=U3 本体，**解冻后首刀**）+2-2..2-9 凑批（合计 95-115ms/跑）；per-layer 上提束刀（a/b/d/e/f 五件，0.9-1.6ms/eager 步）；慢步 fast-path 双次尝试缓存；假遥测 per_step_allocation_count 接真值；僵尸批删（persistent_batch 413+双 ext 影子 610+page_sparse 15 字段+零消费 env 族+危险旋钮 ABLATE_FORCE_FASTPATH/Z_RRP_PROBE）；profile 15 通道合并（−800-1200 行）；三巨型函数拆分；seq_lens 七载体冻结文档。

### D. EXT/triton 审计未修项
- **ext 高危 #2**：accum meta 列宽 CHECK ≥8→≥10（一行，ext 重编窗）；run_soft_nms shm 上限+launch check；cross_head env 非法值 raise；page_size==0 守卫；姊妹文件（soft_nms/cross_head 独立版）退役或同步修（int64/isfinite/warp-align）；**triton A1 长 ctx 雷**（K>65536 LSE 静默截断，128k 即踩——**随 #17 死件删除自动消灭**，若删前要跑 >64k 务必先修）；A3 i32 偏移 i64 化；F2 stride_pad 静默修复改 fail-fast；bounds_kernel C++ 零 CHECK 补齐；rrp_bind 7 数组长度验证；env 判定 C++/python 口径统一；validate_reduce_group 双真源（单一真源手术）。

### E. 正确性/取证未闭环项
- BUG-A2（抢占 resume 绕 bootstrap 门，serve 场取证后修）；U1 static_guard 单点慢步定性（一轮打印）；U2 commit 步内落点分布（决定 epoch 门能否升强制）；U5 kv_len 恒定性计数（决定刀R 行 1/5 联动档位）；audit ② interval 残票 1 步延迟（判无需修留档）；G6 后半（SET-DIVERGENCE 测试参照刷新）。

### F. triton 下线（#17，用户指令）
- P0 死件批（alpha_selector_kernel 整删+attn_reference+alpha_fair 两死 scorer+mixin 死 import+req_meta_flag_codec 搬家）——判据=collect+黄金+判速；P1=pack 族迁 CUDA C++（生产主链，128µs C++ 窗实证可行）。

### G. 发布仓义务
~14 笔待授权同步+**prebuilt .so 必须重打**（三 env 默认 ON）+对端两 .patch.new。

---

# 交接补遗 v3（07-11 晚场：与对端 mixpage kernel 场并行窗，零冲突推进 #17/#12/组X 静态审计）

## §8 本场推进账（main 未动一行代码；施工全在隔离 worktree；GPU 零占用）

### 8.1 组X 臂体案：§6.4 四假设全部静态排除，§6.5 配方修订（判定书=`SFI_ARM_CASE_STATIC_AUDIT_2026-07-11.md`）
- 树锚定：`git diff 88a8072..876ffa1` 在六关键文件（metadata_builder/rrp_row_table_manager/resolved_row_ptr_arena/mixed_page_cudagraph_replay/compact_mixed_page_route/plan_builder）=空 → 结论直接适用 J1 案发树。
- **假设 3（ready event count 语义）排除**：计数器=树上老件（9781ada 07-02 引入非臂新造）；臂 mode=SIG_RETURN→fast-attach 门（:2002-2007 只认 STEADY/PBD）必拒→`_should_bind_...for_update` :847 兜底 return True→count 恒 1（且卡死 4 入口早退+same-page :5013，那些路径 count=0 必炸 fail-close=断言非空洞）；record 与 count 同源（:5555→manager:142）臂上恒真、在当前流（:379-381，缓存按 raw ptr 取真值）、位于行表写（:5426）之后；arena 全文件零 side-stream。「重放流不等 H2D」静态不可构造。
- **假设 1（manager layout 影子基准）强排除**：`_row_table_layout_by_row` 唯一写点=`_store`（:860），update() 四非 HIT 分支必调；触发步 capture 行落 full-KV 分支（plan builder :318-322，valid=0）→七元组巨变→dirty 数学铁定→必重铺。**更根本：全量路径重铺=全批布尔（:5408-5441，delta_rows 只作开关不控子集）→「漏部分行」的 6/8 形态结构上无法产生**。验证钉=print `rrp_update.kind`（预期 PAGE_BOUNDARY_DELTA）。
- **假设 2（bind 序）排除+★§6.4-2 勘误**：慢路径真身序=plan（:6360，含 :6140 就地新建 bound_meta）→xlayer（:6979）→repack（:7432）→SDD（:7508）→**bind（:7562）**——bind 本在 repack 之后，臂 GPU 写序与慢路径**完全相同**；bind 内部零切流。「req_meta 先于行表」无成立窗口。
- **新头号嫌疑=重铺内容/行映射层**：bind_production_row_table 写的内容 vs kernel 消费 descriptor 的行映射（slot/canonical_row_index/per-head 展开）——六嫌疑+四假设只盖了「铺不铺/序对不对」，没盖「铺进去的内容按行映射对不对」。
- 配方修订：**§6.5-3 撤除**（本审计闭环）；§6.5-2 降级搭车打印；**§6.5-1 升级为决定性首跑**（dump 面=row_table_i32+arena_batch_seqused+descriptor_i32 六行+q_layout_key+rrp_update.kind+use_compact/row_mode/short_dense 三 by_row 载体+layout 影子 capture 行；强制 refuse 对照同点位逐位 diff；新增 min 封顶相位判读=token vs 页粒度分辨，:336-339）。预算 ~1-2 GPU 轮。
- **★取证弹药已制备成品=主树 `scratchpad/arm_forensics/` 6 件**（arm_dump.patch 608 行+force_refuse_dump.patch+arm_dump_diff.py 纯 numpy 对拍+test_arm_forensics_cpu.py 假数据单测全绿+RUNBOOK_ARM_FORENSICS.md+lite_arm_body.diff 副本）。apply 链三关 `git apply --check` 全 CLEAN 于 876ffa1（**臂体 diff 零漂移无需 rebase**）；**★apply 顺序严格 ①lite_arm_body ②arm_dump ③force_refuse**（③在缺②树上 --check 也过但运行时 NameError=已录坑）。强拒点比原配方更准：admit 九关全过后、臂第一笔状态写之前→强拒发=绿基线轨迹，8-hash 应回 MATCH 自证；慢路径 :7562 bind 后同函数 dump `_slow` 后缀=同点位，且顺带产出触发步 layout 影子（§6.5-2 搭车不占轮）。dump 永不 raise（故障写 .FAILED.txt 不改变取证形态）；`torch.cuda.synchronize` 仅 env 开启时发生。**★发车形态勘误：J1/repro=GPU2 bs8x12k 判速档（8 请求批,与 6/8 错行自洽），非黄金 bs2**；两发均不设 LITE_SIG_RETURN_ASSERT（案发同款）；首对 step≈111=金标准；取证轮 tps 不作判速（sync 失真）。判读决策树 10 形态→根因层映射=RUNBOOK §6。
- 防重走：canonical_real_kv_len 口径差候补根因已被 :6140-6166 自证击落（全程录判定书 §4a）；臂 `layer_logf_enable=None`→pack ones+kernel AND=无害（§4c）。

### 8.2 #17 P0 死件批：施工完成待 GPU 判据（worktree 分支 commit `40e4ca7`，+14/−534，未合 main）
- ★**上场「alpha_selector_kernel 整删」定谳被活性复核推翻**：bounds 族 5 函数=生产 C++ ext 三件套（selector_log_s/soft_nms/cross_head）对拍测试的 triton oracle（test_selector_log_s_ext:165/268+test_key_norms_fp16_contract:46 活引用）；attn_reference=FA3 native 5 测试的 dense oracle 同 KEEP。**连带：A1 长 ctx 雷（K>65536 LSE 静默截断）未随删消灭，>64k 跑前必修（§D 待办升级）**。
- 实删=alpha_fair 两死 scorer 整链（502→50 行，含死 helper 链+死旋钮 VLLM_SPARSE_ALPHA_ROWS1_FUSE/VLLM_SPARSE_COMPILE_SELECTOR+apply_cross_head_mutex 死件）+mixin 死 import（:42-46 三符号）+`__init__` 两死导出+tp_skip 测试 3；codec 搬家 triton_kernel→utils/（4 引用改齐，行为零变）；alpha_fair 与 triton 完全解耦。双树惯例考证=历次删码批只动 release/（dev/ 自 07-03 冻结）。
- collect 3000→2997 逐项对账（★勘误：本档 §0 的 collect 3016=旧数，876ffa1 实测 3000）。
- ★发现基线红（非本批引入）：`test_req_meta_pack_sink_validation` 断言锚 `_cached_compact_only_stride_pack` 已于 39e0e33（07-08 TRITON-LINE-RETIRED）删除=HEAD 该测试必红；pack 域=对端施工域，留账待收。
- 待 GPU 窗：黄金 bs2+判速档 bs8x12k+counts 三判据后方可合 main。

### 8.3 #12 判据瘦身：前置核查完成+设计在案+verdict-only 实施中（设计稿=`SFI_JUDGE_DIET_DESIGN_2026-07-11.md`）
- ★**新方向 #2 前提勘误**：「正确性判据全在 speed child 产物」不成立——route_proof/counts/fallback/producer_gate 四类判据数据源=diag child 的 `${TAG}_route.jsonl`（bench :3167 构建/:3201 spawn，无条件跑，判速档实测 46.4s=单发 100.5s 的 46%），裸砍必红（route_proof ≥7 missing reasons+producer_gate continuous_refresh_reqs_missing）。
- 可行解=verdict-only 下 **speed child 开 route trace+hook profile**（基建全现成：env 定义 BENCH:81/启用 :3085/emitter 门控与 child 身份无关）；零 GPU 实证=speed/diag child outputs 三样本族逐位一致（含 229 崩案都错得一致）+counts 锚 {sentence:30,interval:4} 平移成立（diag trace 实测正是 {30,4}=warmup17+measured17 全文件计数）。**连带否决新方向 b 刀「跳 warmup」（counts 锚会砍成 {15,2}）**。
- 设计=--verdict-only 默认 False+run_speed.sh VERDICT_ONLY env，~45 行 9 处，默认路径逐语句等价（对端零影响）；跑长 MAX_NEW=256 不减（hash/counts 锚全是跑长函数）。收益=判速单发 100.5→54s，迭代 15→5-7min。
- **实施完成=worktree commit `d3575a1`**（bench+run_speed.sh 两文件 +128/−52，未合 main）：逐 hunk 默认等价论证在案（判读换源三元默认臂=原对象；exit=`diag_result is None or 原两条件`；VERDICT_ARGS 空数组照 TP_ARGS 模式）；CPU 四项全绿（argparse 合同 11/11+collect 3000 不变+fail-fast 5 拒斥+stub 编排闭环 33/33，加菜=真实 diag trace 作 speed trace 夹具跑通换源判读链+counts {30,4} 平移实证）。施工中新发现三件：①refresh_profile 是设计稿未列的陈旧源（`_read_refresh_profile` 无条件读旧文件漏进 producer_gate）已随手封；②`--dry-run` 在 gate-D 模式不可用（BENCH:8899 dry_run 分支在 legacy 路径内，gate-D 下会真发车=坑）；③默认路径唯一可观测变化=summary 纯加两溯源键（verdict_only/route_summary_source，判据器逐键读不受影响；如需字节级零变可一行收窄）。
- 风险 9 项：CPU 4 项已闭；**GPU 标定场 ~20min 闭 5**（trace 观察者税 AB（估 0.2-0.6%→verdict-only tps 只作方向判，定谳保持全量形态）/speed_route 首实跑三对拍/phase_summaries 完备性/黄金档 producer_gate/落树默认形态自证）。

### 8.4 本场并行纪律执行+待 GPU 窗总清单
- 对端 kernel 场施工面=外层老仓 hopper/*+patches/*（M 一片）+GPU0（EngineCore 36.5G）；主树工作区本场全程干净。我方零 GPU、零共享工作区代码触碰、#14 launch_template 刀主动跳过（正撞对端 L1/L2 launch 瘦身面）。
- **待 GPU 窗清单（凑一个窗口批量收）**：①#17 判据三件（黄金+判速+counts）→合 main；②verdict-only 标定场 5 项（~20min）→合 main；③组X 取证 2 发（arm_forensics 弹药已备，~1-2 GPU 轮，决定性）。**窗口编排成品=`scratchpad/GPU_WINDOW_BATTLE_PLAN_2026-07-11.md`**（第 0 步健康自证含 FA .so 形态确认；窗口=独占主树工作区+卡）。第二批推进中：#16 僵尸批删（worktree 施工）+#9 BUG-A2 取证弹药+#8 S1 sync 探针弹药。
- 三笔待合施工的分支坐标（worktree 目录即使被清，分支 ref 在主仓可达；**均基于 876ffa1**）：#17=`worktree-agent-acd28c35f976c62d0` @40e4ca7；verdict-only=`worktree-agent-a330c42b9f1735601` @d3575a1；#16=`task16-zombie-batch` @2863ea5（净 −1440：persistent_batch 整删+双 ext 影子退役（cross_head/soft_nms 参考臂,EXT 审计档双证对照失真）+零消费 env 四件；collect 3000→2998；与 #17 merge-tree 预演零冲突；page_sparse 15 字段+`resolved_sliding_window`+`compact_eligible_rows` 跳过留账；Z_RRP_PROBE 留账且实measured 每 steady 步一次活 env 读=热路径税；ABLATE_FORCE_FASTPATH 删除草案在报告不落刀（:3949/:4040 对端 L 刀热区）；pack_sink_validation 换锚草案在案=测试不退休换第三对覆盖）。本场主树仅文档改动（本档 §8+两新判定书/设计稿 md+scratchpad/arm_forensics/+GPU_WINDOW_BATTLE_PLAN），未 commit，待用户确认后落文档批。
- **★07-11 晚追记：对端 K1 已落 main=`af0ed77`（kind4 pin 条件门+CAP-4+J2 资源门，其注明 e2e 判据接续跑）——我方三分支判据+合入前先 rebase 到对端判据绿后 HEAD；组X 取证保持 876ffa1 案发基座。窗口编排已同步更新（scratchpad/GPU_WINDOW_BATTLE_PLAN_2026-07-11.md 顶部注记）。**
- **★再追记：对端判据已全绿收案 `02847f5`（bs8 判速 609.3 带内 8-hash 逐位不动/黄金 bs2 +51%/★黄金 bs2 换锚拍板件待用户=auto 新锚 {636fb032,44e80946}）+方向定稿 `2e8643e`。我方 rebase 目标=对端稳定 HEAD；判据窗黄金锚以用户拍板为准（拍板前可用 pin1 对照复现旧锚的口径自证）。**

### 8.5 第二批（额度放行后）：#9 BUG-A2 定罪+#8 S1 归因闭环（弹药均成品化）

**#9 BUG-A2＝从"待取证"升格"已定罪"**（复核书=`scratchpad/bug_a2_forensics/A2_STATIC_RECHECK.md`，5 件弹药同目录）：
- 三条触发臂已在 CPU 沙盒逐一复现（vllm stub 无 GPU）；触发链=v1 抢占（scheduler:956-977 回 waiting）→resume 重算窗 `_reset_request_sparse_state_for_resume`（vllm_sparse_patch:2192-2246）只清压缩态+bootstrap_done、**不清票/intent/lease/post_bridge due/last_decode_refresh_step**，而 bootstrap 材料化四层门全部只看 bootstrap_pending（重算窗恒 False）→残票/陈旧时钟/intent/lease/ride-along 五路把 refresh 世代物化进"压缩态刚清零+prefill 半途"的行。
- 后果=crash 级两 raise 形态（wait_decider:749-758 纵深断言+FORCE_NOW 不变量 :4795-4809 已 CPU 复现原文）+最坏静默错值窗（世代 publish 抢先重 finalize）；**A1 修零缓解 A2 反而略加宽**（残留 CROSSED 票连 gap 都拦不住）；审计 §B 行号勘误=A1 修前行号，现行 :4419/:4494/:4580/:4795。
- **★bench 版取证配方可行（推翻"serve 才打得到"）**：`KVB=$((10*1024*1024*1024)) MAX_NEW=1024 REFRESH_INTERVAL=32 … bs8x12k`（满配需求 ≈18.2GiB 压到 10GiB=缺口 45%→抢占循环；run_speed.sh:64-109 自带"mid-run prefill recompute, warns never blocks"书证；单卡半小时）；serve 版配方备用（07-03 注入配方基底+KV 钉小）。
- 修复草案 `A2_FIX_DRAFT.diff`=[A2-RESUME-TICKET-RESET] 清弹药本体（票+intent+lease+post_bridge_due=-1+**last_decode_refresh_step=-1** 由首-decode 锚自动重锚），无 fallback；**未修树合同测试 3 红→修后 3 绿+邻接 6 触发面 20 过**；`scheduled_*/inflight_*` 有意不清（纵深绊线+读侧 dense 闸；"抢占时已在飞世代"隔代子案=独立第五路径，探针快照带 scheduled_* 字段供决策树 4a 定谳）。probe→fix 叠加序 CLEAN。

**#8 S1＝主体归因零 GPU 闭环（用 s7base kineto 留档 152MB 完成 API 级归因）**（弹药=`scratchpad/s1_forensics/`）：
- **孤洞=稳态期唯一一次 writer-graph 三 chunk 捕获步**：`refresh_rebuild_mixin.py:2381` 捕获前置 `refresh_stream.synchronize()`（[ASYNC-CAPTURE B-1] 手动 capture），API 指纹=WaitEvent×18→BeginCapture→1×gather kernel→EndCapture→Instantiate；全 trace BeginCapture 仅 6 次、decode 段 3 次全落洞内；全窗 245 个 ≥1ms sync 中**唯一与主流空闲 100% 重叠的就是这三连**（证据包 hole_microscope/sync_census 同目录）。
- 真刀单口径勘误两处：17.98ms=邻缝合并口径（实 15.8ms@rel 7.588s）；三连真实时序=3.131→0.545→1.008ms。**收益口径修正：三 sync 本体仅 4.68ms，S1 上界=整洞 ~16ms×复发次数**（洞其余=三条 selector 链串行排空+capture host 工作）。
- **Z_RRP_PROBE 考证：与 S1 无关**（7.9ms plateau 旧案残留，全仓唯一消费=metadata_builder 自身，且每 steady 步一次活 env 读=热路径税）——**删除放行**，#16 留账收编入下批小刀。"z 探针"应读作"env 门控打点手法"泛称。
- 弹药=S1_PROBE.patch（876ffa1+af0ed77 双锚 CLEAN；monkeypatch 三族显式 sync 计时+浅栈、`set_sync_debug_mode(1)`+warnings 钩拿隐式 sync 栈不重编 ext；默认关=零热路径新增）+RUNBOOK（五叉决策树+撤针配方+下一步刀三案 K-a 排空收窄/K-b 预捕获/K-c pin 化——**K-b 设计前必读 P3 prewarm 否决收案防重走同类坑**）。GPU 只剩确证+复发计数：单发 3-5min。
- 附带发现（非 S1 入档）：mb.misc 内 pageable H2D `aten::to` ×14/窗（5-14ms/个）=候选 pin 化小刀（行级待探针定位，静态候选 metadata_builder:4444/4446）；nonzero/index 巨 sync×5+gen8 入口 196.8ms event sync 均 GPU-busy 覆盖=免费不立刀。

### 8.6 第三批（用户令稳步推进+避 kernel 冲突；P-0 换锚拍板落档 f60e0f6）

**换锚 P-0＝用户拍板选项 A 已全链落档**（K1 终账 §3 拍板行+两约束、主入口头注、作战单、MEMORY；commit `f60e0f6`）。

**三精度/换锚拍板件思路陈述已呈用户并入仓=`SFI_PRECISION_KNIVES_BRIEF_2026-07-11.md`**（selected_k 降档/Full 刀 score-only/触发密度;考证两大新事实:①生产 k=1536 与 min_gap=24 均无精度背书,已有精度数据全不在判速形态格点上;②**★Full 刀量级或被大幅低估——capture 步实测超额 ≈16.5ms/步×19 步≈毛 +9%**（刀单 +2-3% 疑漏整步 eager 税;s7base trace CPU 重分析,分解=半天 microbench,分解结果决定全场刀序）。三刀全改锚→合一个换锚窗口;精度轴=selected_k×密度联合 LongBench campaign 设计在 brief §联合策略）。

**#15 per-layer 上提＝全禁区裁决,零施工留协调**：五件（a/b/d/e/f）行级亲证后 100% 落对端禁区（patch_installer/kind4 launch 路径/step 级校验缓存域）——#15 整体并入对端 SPLIT-ROOT 后同域刀单。副产物两实锤：①五件等价论证全录（agent 输出;d 件=PI:336 缓存版一行 import 级,对端拆 PI 时顺手兑现）；②**e 件加码:`_RRP_GRAPH_DONE_EVTS` 全仓零 wait 消费点=36 次/步 record 纯死开销,可整删**（真 WAR fence=_RRP_WAR_FENCE_EVT 另一套;比考古档"步末一次"更彻底;归对端 PI 窗,顺带根治 wrapper 内 except-pass 吞错）。

**小件凑批＝第四施工分支 `worktree-agent-ab447daa1395d916f`@`0615654`**（4 文件 +106/−22,collect 3000 不变）：①假遥测 per_step_allocation_count 接真值（13 自增点 branch 粒度完备;快路径硬编 0=结构性真值已扫描亲证;Gate C 消费者活故不删）；②slot_by_row 跨 rank 合同一行（tp_contract 新 `ensure_tp_slot_by_row`:覆盖+非负+步内单射 fail-fast;落点 step_context_worker 绕开 patch_installer）；③G6 后半 SET-DIVERGENCE 参照刷新（`_ref_old_path` 补 sentinel mask,发散测试改判 sentinel_agrees 逐位相等,12 passed）。**附带 finding 留账**：四个同型假遥测兄弟（old_cache_probe/adapter_invocation/steady_delta_d2h_sync/implicit_sync 恒 0 零自增）+thin_builder_state:131 死字段+**`test_prepare_inputs_req_index_failfast_contract` 源锚已漂移=守门测试失效**（对端界文件 patch_installer,留协调修锚）。与 #16 同文件相邻行（合并双保留）,与对端两界零触碰。fast-path 双次尝试缓存件明确未动（撞对端 L1/L2 域,定位现状=:5817/:6111 已录）。

**ext 重编窗批＝九件 patch 弹药成品化,主树 `scratchpad/ext_batch/`**（patches/p1..p9 共 ~701 diff 行+EXT_REBUILD_RUNBOOK.md;基线 3c0b706;九连 apply CLEAN+py_compile 全绿+C++ 括号平衡;**编译验证待重编窗**）：p1 高危#2 accum 列宽 ≥10+负测/p2 soft_nms shm+cross_head env 枚举 raise/p3 page_size 守卫×4/p4 warp-align+两 C++ env gate 严格化+bounds_kernel CHECK/p5 rrp_bind 七数组长度合同/p6 validate_reduce_group 单一真源化/p7 **A1 长 ctx 雷**(K>65536 raise)+A2 rows1 同款/p8 A3 i64 偏移/p9 F2 哨兵语义+F3 pack 合同。作废项核实=姊妹修（被 #16 退役取代）+高危#1（已在树）;F4 降级留账。**重编窗 ~1-1.5h**（★prebuilt 直载雷:load_prebuilt 存 .so 即 dlopen 不比源,须清 torch_extensions 四目录）;4 设计选择点在 runbook §6（A1 案A已落/F2 哨兵/rrp_bind >= vs ==/env strict 覆盖面）。与 #16/#17 merge-tree 零冲突。

**待 GPU 窗清单增补**（并入 8.4/作战单）：④小件批判据+Gate C 实跑（same_page 行计数首验）+slot 合同 TP 场实弹;⑤ext 批重编窗 1-1.5h（凑 #17/#16/verdict-only 同窗,黄金锚用已拍板新锚）;⑥Full 刀量级分解 microbench 半天（拍板前置,用户已知悉）。**我方施工分支现四个**：#17@40e4ca7/#16@2863ea5/verdict-only@d3575a1/小件批@0615654,全部基于 876ffa1 或 3c0b706,合入前 rebase 对端稳定 HEAD。

### 8.7 第四批：远端 b91f77e 反馈四开放项全定谳+速度时间账（回复稿=`SFI_REMOTE_TP8_FEEDBACK_REPLY_2026-07-11.md` 可直接转发）

远端档=`b91f77e_三案根修_测速反馈_给作者.md`（案 A parity 已解除+倒挂翻正 0.765→1.068×;四开放项交回）。三份独立判定书全闭环：

- **案 B＝我方判读口径 bug（8× 观测通胀），产品达标**：TP8 八 worker 各自把同一世代事件写进同一 route trace（发射点 PI:8959 无 rank 门+聚合器 phase1:1530 不按 pid 归一+expected 公式单 controller 口径）→一切计数 8× 通胀;**铁证=replay_refresh 16376/8=2047≈decode 步 2048**;÷8 后比值 bs32=0.93/1.10/0.89、bs16=1.09 全落"≈1.0 健康带"。上轮 5.66×÷8=0.71（缺拍+escape 工况）→本轮 1.09=**门修后触发线恢复满额的证据,"升高"=修好了**（门期间 interval 连票都不落,intents 计数点在世代提交,门不可能造成通胀——远端"门推后触发→intents 更多"方向不成立）。joined=0=设计内（同拍工况"到点判定"与"join 放行"读同一对时钟,结构性无人可搭;错峰配方两种已给远端）。真 bug 排查：无 intent 风暴（三层抑制:gap 蕴含/no-restamp/票 consumed）;**sentence×interval 四面安全在门修后仍成立（用户铁律复核过）**;audit② 维持无需修;无害小疵=混编世代 reason 单值互染 ±10%（解释 rep 摆动,记档）。
- **确定性＝承诺域外+作者侧真嫌疑一项**：run 间 bitwise 只在 bs2long-cap128(TP1) 锚工况承诺;TP-DET 族全部承诺的是 rank 间决策一致。dense 位确定豁免共享栈→首分叉锁死 sparse 组件;**头号嫌疑=selector topk GPU 非确定**（k 界并列 atomic 竞争+`sorted=false` 输出序,pipeline_ext:2546/2601→pack 序直通 rebuild:2431（物理序重排 env 默认关,constants:141）→FA fp 累加序低位漂→贪心近平局翻转→雪崩;曝光 ∝ 世代×步数,bs2 锚工况看不见）。**refresh 兑现时点 async 假说被推翻**（决策链全 host 决定论化,时序进不了决策——早先直觉口径已勘误）。收口候选=决定论化 tie-break/规范化 pack 序（会换锚,拍板件池）。R1-R6 取证配方已给远端（首件=零 GPU 最长公共前缀）。
- **rrp_count_mismatch＝reason 名误读**：该检查=speed child **单 run 测量窗内**路由形状自洽账（python 可见 forward 总数 vs RRP 路由数,graph replay 步不计入,install.py:119-153 计数器+bench:1756-1886 两分支）,非 rep/rank 间比较;r2-only 触发=run 间发散下游指纹（步组成不同）,非阻断;已请远端回贴 proof 四计数复核分支②量级。
- **flash 探针＝真缺口已修 [TRANSFORMERS-PDM-SEED]＝第五施工分支** `worktree-agent-adf4add7083fe0ede`@`3e40e1c`（install.py+两测试,+314/−11）:亲证缺口比远端定位更大——**7 处裸下标 2 把 key,modeling 兼容矩阵 lambda(:78/:97/:105) 直查 dict 不经任何探针函数=函数包裹永远盖不住,播种共享 dict 层是唯一正解**;修=原地播种非空哑分发（哑名避开 flash-attn/-3/-4;真分发不覆写;4.x 自适应跳过;except 窄化 (KeyError,IndexError) 第二层保留）;**真 transformers 5.6.2 wheel 双幕亲证**（无播种逐位复现远端 KeyError/播种后全 False）;合同 2→8 全绿+端到端 dummy 模块双形态;旧 guard 注释勘误（PackageNotFoundError 非 KeyError 子类,防线有效注释错）。发布仓 §5 考证=官方父 runner 随仓在,直呼 worker 才撞守卫→README 写自验配方+锚值（下批同步）。
- **速度时间账定谳（用户"TP8 速度不合理"）**：dense 55ms×2048=112.6s 精确自洽其 582tps;sparse 稳态 28.7s vs 总 105.5s→**世代税 77s=73%**（per-pid 修正=~404 拍×~190ms/拍）;稳态步 14ms=dense 的 1/4=sparse 稳态面全场唯一健康。四层分解=世代税(主线刀靶)+4B@TP8 框架税(dense 自身 50ms 通信/host)+**配方项 custom AR**（考证=expandable_segments×AR 互斥预检 PI:3893,A800 可关 ES 开 AR 做 AB=可能白捡)+版本差(积压批几个点)。
- **观测面修在途**：route trace per-pid 归一（第六施工 agent,聚合断言 fail-loud,TP1 逐位不变硬约束）。
- **发布仓同步批（待授权）增补件**：flash 播种修+per-pid 观测修+README 自验配方与新锚值+积压批全家+prebuilt .so 重打;推送后远端按回复稿 §6 六项重测。

### 8.8 修复批终态（用户令"先修完再回复远端,远端=纯测速机器"——纪律已入 memory 必守区;回复稿已改写 v2=修完再发口径,删除全部"请远端取证/贴数据/手工换算"项）

三件修复全部施工完成（皆 worktree 分支,未合 main）:
1. **flash 播种修**=分支五 `worktree-agent-adf4add7083fe0ede`@`3e40e1c`（§8.7 已录）。
2. **per-pid 观测修**=分支六 `worktree-agent-a3f450329b229a6bd`@`e1e4473`（4 文件 +573/−30）:**选案 2=聚合咽喉一处归一**（否决分文件案:消费者面 ≥15 处迁移+truncate 陈旧污染新风险;`_route_summary` 是 e2e/official 共同导入的唯一聚合点,counts/route_proof/replay 门全从它取值）;≥2 pid 组按签名断言一致取首组,不一致 fail-loud 绝不静默平均;撕裂行如实计数+容忍上界(≤2×坏行);**TP1 逐位不变双证**（旧 41 键机器对拍 diff=0+单 pid 原对象 `is` 直通）;新合同 11/11+消费者面 445 passed 零新增红;与 verdict-only/flash 分支 merge 预演干净。远端下轮重测=天然验证场（预期 replay 回 ~2047 级+pid_count=8;fail-loud 抛错=workers 真发散=新案信号）。
3. **topk 非确定收口 L1**=分支七 `worktree-agent-aa2b00c7f018592c2`@`9732635`（12 文件 +321/−164,**★换锚件与精度刀凑同窗**）:落地=selection 生产点新 `canonicalize_selected_indices_pack_order`（升序+-1 移尾+stable）,writer 三调用点+tracking 全消费面收规范序;**否决 REBUILD_PHYSICAL_BLOCK_SORT 转正**（物理键=allocator 状态函数,锚随 gpu_mem_util/抢占/prefix-cache 漂;逻辑 index 键=选中集合纯函数,锚只依赖(模型,输入,选择)——且旧实现带 phase 门+silent-skip fail-open 枝）,旋钮全链删零残留;**★附带根修一个选集级非确定病灶**=gather kernel 前缀消费 persist_len 槽且 -1 不跳过打 ordinal 填充（短行 regime 有效 pick 落前缀外被丢——规范化后有效 pick 恒压前缀,一并消除）;决定论单测 6/6+无并列行为最小变化证明+collect 对账。**L2（k 界并列 atomic 竞争=选择集非确定）待窗**:主案=边界并列带二段修复（分数零扰动,+2 全 K pass）,对照案 sort 截 k（+7.2GB 瞬时=主要否决点）,microbench 配方在报告;GPU 窗七项=选集对拍/L2 定谳/换锚三件重立/判速 AB/TP8 原案复验/触发交互面复核/.so 无需重打（纯 py）。与小件批一处 docstring 文本冲突（test_fixed_shape_topk 同段 prose,语义正交后合者并）。
4. **★对端 SPLIT-ROOT+L1+K2 已落 main（§10,判速 611→671=+10%,refresh 窗 writer −53.8%,8-hash 新锚已拍板,.so=d5a692 形态成套铁律）——推送批量级预期上修**:writer −53.8% 直接砍世代税主项,64k 长 ctx 形态收益更大;远端第一轮推送=积压+修复三件+SPLIT-ROOT 家族+.so d5a692 重打,预期速度提升 10%+ 级（非此前"几个点"口径）。我方七分支 rebase 目标=54a74af 后对端稳定 HEAD;判据窗用新锚（判速 671-673 带+8-hash 新组+黄金 {636fb032,44e80946}）。

### 8.9 ★收针定序：与 kernel 线的三分法协调（用户拍板 07-12;**新 session 推进从此节进入**,详单=总作战单 v2）

**协调原则（用户确认）：不全等 kernel 线,按耦合度三分**——对端 N4+N1 在途且锚基线随其刀频繁更换(07-11 已换两次),盲目合入=反复 rebase 重验;但零耦合件停等=纯浪费。

**①零耦合件=立即可做（"小取证窗",半天,空闲卡,零碰对端）——新 session 首务**:
1. **组X 取证 2 发**（弹药=scratchpad/arm_forensics/,基座钉死 876ffa1 案发树=对端动 main 无影响;apply 序①②③严格;J1 形态=bs8x12k 非 bs2;发 B 强拒 8-hash 应回 MATCH 自证;对拍→RUNBOOK §6 决策树定根因层→回填本档 §6→**Lite v1.1 解锁**）;
2. **A2 bench 取证+fix 正式落地**（弹药=scratchpad/bug_a2_forensics/,KVB=10GiB 配方半小时;fix draft 合同已 3 红→3 绿,实锤即落）;
3. **S1 确证单发**（3-5min,弹药=scratchpad/s1_forensics/）;
4. **Full 刀分解 microbench**（半天,kernel-only;**必须在 d5a692 现树上量**——SPLIT-ROOT 已改 split 行为,capture eager 税要量新形态;对端再动 kernel 重跑仅 10min 级,不构成等待理由）;
5. **密度速度轴 AB**（min_gap {24,32,48} 同卡交替,现树;AB 腿判读只看 tps+组内自洽,勿动生产默认值）。
→ 产出:组X 破案+A2 闭案+S1 立刀数据+**两份拍板数据→呈用户拍精度刀**。

**②合入类=等对端 N4/N1 落完的稳定 HEAD,一次收账（"合入窗"）**:七分支一次 rebase+批量判据(verdict-only 模式 54s/发)+合 main;判据用 §10 新锚(671-673 带+8-hash 新组+黄金不变)。**唯一例外=verdict-only 分支(d3575a1)可先合**——判据 harness 与 kernel 零耦合,先合则两条线判据都提速一倍。

**③必须等的**:topk-L1 换锚件(@9732635)=并入下一个自然换锚窗（对端下次换锚或精度刀落地凑一窗,**勿单独第三次换锚**）;对端域协调件(#15 per-layer+EVTS 整删/#14/fast-path 缓存)=等 SPLIT-ROOT 域收针;**推送轮=等对端本阶段收针**（多带 N4/N1 两刀,远端少测一轮;★推送需用户授权,push 前再确认;推送批内容清单=总作战单 v2）。

**新 session 接手路径**:MEMORY 索引首条 → 本档头注两行（最新锚+.so d5a692 铁律）→ **本节** → `scratchpad/GPU_WINDOW_BATTLE_PLAN_2026-07-11.md` v2（逐步详单+推送批清单）。坑速览:发车前 git log 锚定+判读查 run_provenance.git_head;判速 <±1tps 单发不定谳;PYTHON=vllm019-cu126 必带;命令绝对路径防 cwd 坑;ext JIT 用独立 TORCH_EXTENSIONS_DIR;外部进程会抢卡=预检与发车同命令原子。

---

# 交接补遗 v4(07-11 下午场:K1 占用感知 split 门落地全绿+kind4 方向定稿)

## §9 K1 场推进账(main 876ffa1 → af0ed77 → 02847f5 → 2e8643e,三 commit 全带判据;树净;.so 未动)

- **K1 已落地并全链收案**:kind4 稳态 RRP 发射点 num_splits 恒 1 → 条件门(b*h_k*2≥num_sm 钉 1,否则 0=auto)。判据:sanitizer 0 错/kernel-only 曲线×2 轮(bs2×sel8k auto 3.9×,bs8×sel4k pin1 快 5.5%)/**bs8x12k 609.3 带内+8-hash 与 CAMPAIGN 历史锚逐位一致**/**bs2x30k AB×2=+34.9%**/**黄金 bs2 +51%**。终账+作业级交接(锚值表/配方/发布仓 delta/下一场序)=**`SFI_K1_SPLIT_GATE_LANDED_2026-07-11.md`(§6=发车所需全部)**。
- **★拍板件 P-0(阻塞黄金档判读口径)**:bs2 档 auto 使黄金 seq1 换锚,新锚 {636fb032, 44e80946}(决定性已证);pin1 一行还原逐位复现旧锚 {636fb032, 0d5f663c}=单变量证明。拍板前黄金判读用双态口径。
- **kind4 下一场方向定稿=`SFI_KIND4_NEXT_OPT_DIRECTIONS_2026-07-11.md`**(5-agent workflow:4 视角+对抗裁判 12 锚点实读):10 刀+3 拍板件+封路扩 7;主刀=SPLIT-ROOT(prepare 离散 makespan argmin+host 同式静态实例选择+删 python 门=删全部机器常数;三实测锚逐点复现;A100 标准档 e2e 零变化诚实声明;收益=中间带 bs3-7/H100 饱和带/可移植性)。施工序=刀1 S0-a+刀2 L0(两个半天取证)→刀3 SPLIT-ROOT→刀4 K2→…;**SPLIT-ROOT 跨 host/kernel 两地盘,施工前须与 host session 协同窗口**。
- **判据基建**:J2 资源门工具+SM80 基线(572 kernels,STACK 直方图零差,负测已验)=K2/K7/half2 重做前置;J1=kernel-only 硬门史上首次真全绿(9 退休 case 契约化+0.4% 速度门 scope 根修:launch_event 模式量的是入口 host 税,历史"绿"从未实跑+默认切 cuda_graph)+CAP-4 split×capture valid 区逐位钉+capcheck 4 处预存崩根修+hotspots --case full 复活。
- **勘误/坑(已入记忆)**:黄金正典配方=one-shot --preset bs2long-cap128(直跑 run_sparse_only=diag 形态,裸 env 必踩"FlashAttention version not detected"假征状);黄金 hash 口径=sha256(str(token_ids))[:8];Step2 供数收紧算术封路勿再开;make_context_corpus.py=run_speed 死引用(bs2x30k 语料再生配方在终账 §6.2);后台 Bash 工具 10 分钟上限(长 AB 逐腿驱动);外部进程抢卡(预检发车同命令原子)。
- 与 §8 并行会话零冲突交叉确认:其三分支将 rebase 至 2e8643e;组X 取证基座维持 876ffa1。

---

# 交接补遗 v5(07-11 晚场:kernel 场三刀落地=判速 +10%+R2/N2 科学基建)

## §10 本场推进账(main 3c0b706→ef8b0e1(L1)→cfb090e(SPLIT-ROOT★)→54a74af(K2);.so=d5a692796c14)

- **判据链换基线(全部用户拍板过)**:bs8x12k 判速 tps 新基线 **671-673 带**(旧 605-612);**8-hash 新锚 {2659f3a1, 33e76599, 58b3b21e, cb34ce25, f0b76039, cc46a168, cf8a3581, e57dc3a4}**(3 根 MOVED;pin1 一行还原逐位复现旧锚=单变量自证;逐请求文本无崩坏已验);黄金 bs2 {636fb032, 44e80946} 不变;J2 SM80 基线换锚 tests/FA3_SM80_RESOURCE_BASELINE_**a97b7e9**.json(旧版保留);counts 19 类口径不变。**★成套铁律:patch(域 cap)+FA(argmin+py/C++ 双 guard)+.so 三件必须成套**——旧 .so guard 拒显式 num_splits>1=EngineCore 启动崩「kind4 split is not supported」。
- **SPLIT-ROOT Phase1(主刀,FA 2ae0326)**:prepare 离散 makespan argmin(多波域,3 波帽=S0-a 取证界;波常数 1=校准值如实申报+H100 五档可证伪预测表);host 删 K1 半 SM 门→恒发 tile 无关域上界 cap(cap-dominance 定理+23 万格单测);判速档 dyn=5 census {5:36}。e2e +10.1%(672.88 vs 611.27,主树自证 671.2);**kind4 vs raw native FA3 同 s 税 ≤0.29% 全档=不变量保持**。方向定稿刀3 的「一波帽」被 S0-a 证伪(bs8 饱和档 s3=−15.6%/s5=−27.3%;s2 恰是旧公式唯一解=最差档),设计已按实验修订。
- **S0-a 仪器遗产**:--force-num-splits CLI 拒>1(K1 旋钮只实跑过 {0,1});显式>1 真实语义=cap(flash_api.cpp:936);钉任意 s=metadata 注入仪器(get_scheduler_metadata+sm_margin=-10000,scratchpad s0a_split_pin_runner.py 复用勿重造)。
- **L1(FA 48ae379)**:python 层 memo 三件,T_py 17.8→9.4µs(bare 46.8→33.4);L0 勘误=dense 37.2 是 launch 队列污染值(复测口径 N≤300);C++ 段 21.4µs 内部未分解=NVTX 窗遗留。
- **K2(FA a97b7e9)**:kMaxTilePages 1024 仅 hdim≤128(hdim256 档距 227KB opt-in 顶 2KB=全量会 launch fail,洞已堵);SM80 warp 页表预解析 rider(!Q_in_regs 编译期门);pin1 档 −2.2~2.3%;J2 字面红拍板案例(STACK 仅降+REG 零差=接受+基线换锚)。
- **R2 方向裁决=`SFI_KIND4_DIRECTIONS_R2_2026-07-11.md`**(2 视角+裁判 26 锚实读):N1 prepare 死发射/N3 KS1-REORDER(half2 同型最高危)/N4 split 臂 kStages=2(改判成立:REG=255→该臂本就 1 CTA/SM);D6 降级 30min;K7-Step2 封存;combine 封路维持+κ 闸门;装箱 ABI 封路。
- **N2 四判定**:①H1 深度饥饿成立(attn 879 vs 天花板 1303GB/s;kStages=1 串行装载=绑定结构)→N3/N4 上界 sel4k ≤+50%;②κ=4.9-5.8µs≥4→combine 重过堂但 ≤1.6% e2e;③K6 口径=1.22%/步(argmin 净增税 +1.19µs/发一并收);④**朴素 f 公式废止:SPLIT-ROOT +9.9% 的 7.7pp 缺口=refresh 窗 writer attention −53.8%(550→254µs)=split 类刀对 writer 面收益大于 decode 面**;f_pure=26.9%(纯 decode 刀适用);判速步全貌=attn 26.9%+prepare 1.2%+combine 1.6%+GEMM 双王 67%+idle 2.7%。
- **坑新增**:nsys 2023.4 判速 node trace 确定性崩(→kineto sitecustomize 注入替代,tps 无感);ncu 硬 counter 权限封死(RmProfilingAdminOnly=1);python+C++ 双 guard 面(cap_smoke 拦截在 op 层=py guard 工具盲区,e2e 首发才炸);主树 build .o 缓存不含并行场构建(叠加重编必须从新源);cwd 漏 cd 四连犯→发车配方固化 wrapper 脚本;kernel-only 速度门 auto 形态 7 case 红=splitroot 公式分叉形态效应(非缺陷,归 J1 bench raw 列 argmin 对齐)。
- **在途**:N4+N1 合并施工窗(一次重编两刀分离判据;N4=H1 上界实兑,J2 STACK 漂移即停)。
- **发布仓义务 delta(并入积压)**:L1/SPLIT-ROOT/K2 全部件+新 J2 基线+**prebuilt .so 必须按 d5a692 形态重打**+判据说明锚值更新(判速 8-hash 新锚+tps 新带)。

### §10.1 追记(07-12 凌晨:N1+K6 双合入+N4 撤刀案+两判据环境知情项)

- **main 链更新**:54a74af→(host 场文档批 2081c05/ee9e196/fe4e572)→**d43037e(N1)→b61ee41(K6)**;主树 .so=**b2d28e6d57fb**(N1-only 版;N4-reopen 绿后再叠加重编)。合入后组合形态 capcheck rc=0 自证。
- **N1(FA efae96e)**:prepare 死发射消除(SingleTile 形态 prepare 计数 20→0 教科书级;pin1 计时 −2.11%;J2 per-record 1076 条逐位零差;判速生产档 dyn5 构造性零变化=收益在 s==1 形态)。
- **K6(patch 6b03069)**:scheduler_metadata 单一真源=capture 发射点单发 prepare 捕入 graph、后 35 层共享(★实读勘误:36 发在 capture 发射点每 replay 重放,非 eager);prepare 税 143.6→3.1µs/步=净省 140µs≈1.28%(kineto 生产复拍证明);GPU 微对拍 36 层跨腿逐位+CPU 合同 17/17;SM8x 门(SM90 走旧路径避 memset 反噬)。+死件删除(fcefe81):_RRP_GRAPH_DONE_EVTS 族 record-only 零消费整删 −57 行(真 WAR fence 独立对保留)。
- **★N4 撤刀案(J2 硬门教科书级拦截)**:split 臂直接 kStages=2=172 kernel STACK 3568-4112B(half2 同型全量 acc spill,REG 让位降 238-249)——"上游 kStages=2 无恙"不可外推 REG=255 顶格臂=half2 教训二次应验;revert 在案(a7fcca6),GPU 判据零浪费。**重开已由 N4-reopen 正向路径接管**:预研逐符号实证 Q_in_regs=false(释 32 常驻 regs→ptxas 转投消 spill,生产臂 in-loop LDL 92→9)→kStages=2 后 STACK 40-48B 反低于基线;施工中(122 TU 重编+判据)。
- **★两个判据环境知情项**:①**判速 bs8x12k 8-hash 跨时窗非确定**(同代码同 .so 五腿三组 hash;主树纯净腿自己对锚 4 根 MOVED;归因=topk 选集物理键随 allocator 漂病灶,host 场修复分支 9732635 待其换锚窗)——判读预案=同窗对照腿+构造级判据(kernel-only 逐位 oracle/capcheck/黄金锚不受影响)替代;tps 波动 ±20-40tps 单发不可定谳。②**counts reason 对启动段墙钟相位敏感**(N1 使 sentence 30→29,三腿单变量+hash 逐位闭环=非语义缺陷)——建议判据口径改"类集合+hash 硬门、reason ±1 容差"。
- 在途:N3(kS1 重排,J2 正向迭代中)/N4-reopen 施工/J1 工具账(bench raw 列同 s 对齐);队列=#13 C++ 瘦身批/#14 D6/#15 刀D(触 <0.5% 停止线则封存)/#16 场末终版。
- 发布仓义务再增:N1/K6 delta+.so 按最终叠加形态重打(当前 b2d28e6d,N4-reopen/N3 绿后为准)。

---

# §11（07-12 凌晨场收针指针）窗0 全闭环+合入窗执行+组X 收官
全账=`SFI_WINDOW0_MERGE_BATTLE_REPORT_2026-07-12.md`（战报正式版）。要点：合入批 13 支落 main（总闸绿：黄金逐位+判速 hash 同窗逐位不动）；组X 案五发定谳链收官=E1 descriptor pinned WAR 主线级潜伏洞+R1' 根修双实锤（X2：582 tiny sync=隐性护栏被 K-c 拆开→r3 叠加绿）；AMNESTY 信号合一落地（用户规范）；A2 抢占窗清弹药全判据绿；S1 K-a/K-c 形态腿全绿；Full 刀崩塌 +0.9%（straggler 税已被 SPLIT-ROOT 覆盖）；gap48 反弹=编队拆解相变（交互面审查档）；换锚腿×3 产候选新锚呈拍板。RUNBOOK 三勘误已回填（A2 加压轴=增长比/S1 栈过滤/组X 取证终账）。教训固化=判速机器级独占/worktree 三件套/pkill [x] 转义/孤儿三层清/Monitor 匹配词避启动日志。

### §10.2 终版收账(07-12 晨:kernel 场收兵,十件落地,停止线触发)

**main 终链**:…b61ee41(K6)→1674e81(D6 收案)→bf17ebe(J1 工具账)→4fefd47(#13 C++ 瘦身批)→**d2843c3(N3+N4-reopen 终局)**。**主树 .so 终版=`04cbbf9794fa`**(N1+L2R2+N3+N4-reopen 四刀 122 TU;成套铁律持续有效)。

**本场十件落地总清单**(全部判据链全绿合入):L1(python memo −49%)/SPLIT-ROOT(**判速 +10%=611→671,主力**)/K2(页表双刀)/N1(prepare 死发射)/K6(prepare 36→1=1.28%)/#15 死件删除(−57 行)/J1 工具账(速度门影子对根修)/#13(28 处 const ref+卫生+五段账)/N3(kS1 重排,受益面 −0.2~−1.4%)/N4-reopen(Q_out×kS2,非饱和档 −4.4~−6.9%)。判速档参考读数已进入 686-687 区间(环境噪声下不单发定谳,机械账支持上移)。

**★H1 终审(三段取证链闭环,后场勿重开)**:N2 上界 ≤+50% → N3 重排实兑 2-4% → N4-reopen 深度饱和档 +0.15%=零(wave 间并行遮蔽)、非饱和档 −4.4~−6.9%=真实。结论=**饱和档瓶颈在带宽面,kernel 深水面(重排/深度/combine 融合=刀D 已封存)触停止线收兵**;剩余大空间只在带宽路径(封路区维持)与非 kernel 面。

**判据基线终态**:8-hash 判速锚+黄金锚不变(§10);J2 基线三代并存(ff6fb42/a97b7e9/**b08420e=现役**);kernel-only 速度门=影子对语义(J1 修后,门内=纯结构税);判速 tps 参考带待 topk-L1(host 场)合入后重定(当前环境非确定,686-687 为最新多发参考);counts 口径建议=类集合+hash 硬门、reason ±1 容差(N1 相位案)。

**拍板/授权件呈用户**:①**P-2 lse arena 立项**(供料成立=1.56µs/call 实测,op schema 尾置 optional 参+发布仓 .patch 同批+prebuilt 重打搭车);②**发布仓同步授权**(十件 delta+prebuilt .so 按 04cbbf9794fa 形态重打+J2 新基线+判据说明锚值更新;push 只走 LV-NUS/SFI cuda-kernel);③oracle 腿同 s 化(bs8 失配档数值 oracle 病=现状,触 19 case oracle 契约=另案拍板)。

**新坑七条(本场后半)**:拼装 TU 模块级弃驻留悬崖(单 TU probe 盲区,J2 全量门唯一抓手)/reset 去一被合同击红(fail-closed 合同=每入口 reset 先于失败门)/主树 build .o 缓存 stale(7aaf7d9 时代,控制重链须自证)/worktree e2e 缺语料(从主树 out/ 逐字节拷)/nsys 判速崩+ncu 权限封死(kineto sitecustomize 替代)/session limit 中断=SendMessage 原地续跑无损/run_speed 产物写非原子(瞬时故障一例,复跑即绿)。

**下场序**:①发布仓授权窗(用户);②host 场 topk-L1 换锚窗后判速环境判据恢复;③远端 H100(用户已后置=代码就绪即可:SM90a 门+资源零差+架构中性论证全部在案);④P-2 若批=lse arena 施工(半天+ABI 演进);⑤R2 遗留另案(oracle 同 s 化/irregular 页集对照)。

---

# §12（07-12 上午场）拍板批落档+授权窗执行账

## §12.1 用户拍板记录（07-12 上午，全部授权）

用户令："按交接档继续，授权全部呈报拍板件，把本地能做的全做完。" 逐件裁定：

1. **新锚转正**（战报拍板单①）：8-hash 判速锚正典 **{242dfec6, b18ff00c, 58b3b21e, 2017b63c, f0b76039, cc46a168, d04130be, e57dc3a4}**；counts 新形态 {sentence:15, interval:16}/payloads=612；黄金 {636fb032, 44e80946} 不变。判读口径=类集合+hash 硬门、reason ±1 容差。tps 正式带待独占批。→ 本场已落档（头注★★★行+RUNBOOK 判读五关）。
2. **P1-B 立项**（gap 节流阀上移世代级，gap48 反弹根修+规范直译）：施工停靠分支，**换锚件归下一换锚窗**，不合 main。
3. **gap 生产值 24→32**：已授权；**落地前置=密度轴正式轮数据**（新锚形态独占跑，拍板单原文口径），授权先记账不先改值。
4. **K-a/K-c 判速兑现账**：独占批终判授权；当前机器非独占（外部 EngineCore×2 满载 GPU0/1，load≈10），挂安静窗监控执行。
5. **发布仓推送授权**（=§10.2 件②）：积压全量+合入批 13 支+对端 N1/K6+kernel 场十件+P-2 搭车；prebuilt 义务=SFI ext .so（三 env 默认 ON 形态）+FA3 .so 按 P-2 后终形态重打；判据说明锚值更新随批。
6. **P-2 lse arena 立项**（=§10.2 件①）：施工=fwd_mixed_page op schema 尾置 additive optional lse 出参（ABI 演进）+python 层 arena；数值零影响、黄金 hash 不动；发布仓两 .patch 同批换。
7. **oracle 腿同 s 化**（=§10.2 件③）：kernel-only bench oracle/reference 腿与被测 case 同 num_splits 发射，消 bs8 失配档数值 oracle 病；19 case oracle 契约同批更新。

## §12.2 本场推进账（滚动更新）

- 新锚转正落档 ✓（本节+头注+RUNBOOK 判读五关同步更新）。
- **★host 场并行 session 认领（08:30，勿重复施工）**：①Lite v1.1 臂体返工（#7，armA' 判据配方，GPU7 判速档+GPU5 黄金）②P1-B 施工（interplay 审计"根修草案方向 B"，停靠分支不合 main）③L2 topk 并列主案（边界并列带二段修复，停靠；microbench 用 GPU4）。P-2/oracle 同 s 化/发布仓组装/独占判速批安静窗=按 §12.1 归 kernel 场 session 执行。
- **kernel 场 session 推进（09:40 滚动）**：
  - **oracle 同 s 化 ✓ 落地 commit 3606c95**：correctness reference 按被测 case 的 effective split 重发（cap 语义+launch-state 读回验证，未证 pin 落回 auto 缓存并以 oracle_split_status 曝光）；19 case exit 0 双腿，活 oracle 10/10 same_split 逐位 0.00e+00（含 bs8 irregular 病例档）；**契约基座 4 红全数顺修**（J1 遗留：EXPECTED 案单缺 3 case/ABI 58→59/force 包装两锚）=22/22 全绿。
  - **P-2 lse arena 施工完成+判据 8/9 绿**：fwd_mixed_page/fwd 双 op schema 尾置 `Tensor!? softmax_lse_out=None`（59/38 参）+mha_fwd 分配点 fail-fast 复用+wrapper 级 per-device grow-only arena（丢弃路径 only；grow 保活 retire 列表防 graph 野写）。**全量干净重编 .so=fc8dbbcf8eaa**（主树 build/ 缓存 stale 实锤绕开，122 elf sm_80）。绿：schema 冒烟 59/38 ✓、lse 逐位四腿自证 ✓、RUNBOOK 异常门 ✓（勘误：kind4 splits 预期 0/1=SPLIT-ROOT 语义）、capcheck 12 case ✓、**J2 直方图 614 kernels 逐项同=kernel 零动+编译确定性双自证** ✓、abi_smoke 42 红逐项同基线 ✓、kernel-only 19 case exit 0 ✓、**黄金 bs2 {636fb032,44e80946} 逐位 MATCH** ✓。
  - **★P-2 判速 e2e 红案（二分推进账，11:00 滚动）**：首发红腿（非 vo）sparse never engaged（reqs=0/route 空；vLLM dense 照跑 tps 666=静默未装）。腿 A（旧 .so+旧 py+vo）实质绿=reqs 612+counts 正典+**8-hash 对新锚 8/8 MATCH（新锚跨 session 首次独立复现）**；腿 B（新 .so+旧 py+vo）实质绿=**.so 无罪**。**★二分设计自查：A/B 同时引入 VERDICT_ONLY=confound**——红腿是唯一非 vo 腿；且合入批 13 支的总闸=vo 腿，**非 vo×defer 形态自合入窗后从未验证过**。stderr 全文（12KB 窗）零异常=EngineCore 正常起+capture+退，红腿 child env 独有 `VLLM_SPARSE_DEFER_BOOTSTRAP_PRODUCER=1`（非 vo 才注入的 defer 链）+`bootstrap_full_kv_handoff=False`=defer 交接未发生。**★腿 D4（全量 P-2+vo）=完全绿**：reqs 612+counts {15,16}+8-hash 对新锚 8/8 逐位 MATCH+tps 665.5 → **P-2 判速判据（vo 正典形态）闭环=P-2 无罪定谳，FA commit 4c405dc 落树（新成套铁律=FA 4c405dc+.so fc8dbbcf8eaa）**。历史知情项曾把 `async_producer_writer_missing` 当 vo 判读噪声；**该口径已被 §14 runner fail-closed 审计推翻，现在必须作真红**。中途三坑（D1-D3 三折）：双 session ext 乒乓重编（worktree↔主树切换全量重编+FileBaton 互等）→终解=私有 TORCH_EXTENSIONS_DIR 以本方源预热（dir2，一次性 ~15min，之后全腿秒过）；host 场实际用卡漂移至 GPU3/GPU6（卡位表已失效，发车前 ps 实查代替查表）。
  - **★★红案收案（六腿矩阵终局，判定书=`SFI_P2_SPEEDLEG_REDCASE_VERDICT_2026-07-12.md`）**：E（隔离旧树+非 vo）绿=SPEED RUN OK；**F（案发形态完整复刻：新全套+非 vo）绿=SPEED RUN OK+reqs 612+counts 正典+8-hash 8/8 MATCH** → **红案=一次性环境事故，当代不可复现；P-2/合入批 13 支/非 vo defer 形态三方全部无罪**。事故窗归因（情理链）=案发腿 09:18 落在我方 .so 换装窗后 10 分钟+主 ext 缓存三方 prewarm 竞态首窗（FileBaton 半新半旧部分装载→sparse 施工静默败→vLLM dense 照跑）；与 host 场 r1 撞窗案互为同窗两面。处方三条已入 memory 纪律（互查窗口/私有 ext 缓存/复刻腿定谳法）。
  - **★发布仓 push 完成：`b91f77e..320e5fb` → LV-NUS/SFI cuda-kernel**（授权件⑤兑现）。一笔含：fa3 patch 重生成（base f5bc33c→FA 4c405dc，26 文件=SPLIT-ROOT 家族+K1/K2/N1/K6/N3/N4+J1+#13+P-2）+patches 全量追平（13 支合入批+codec 搬家+persistent_batch 删）+benchmarks/scripts 同步（含 VERDICT_ONLY 通道）+kernel_patches README 的 SM100 知情注记（fa4 patch 待下窗 rebase，SM80/90 不受影响）。远端指引=`REMOTE_TEST_INSTRUCTIONS_2026-07-12.md`（新锚组+重编义务+VERDICT_ONLY 可选）。发布树独立自验豁免推理=patches 逐字同步验证+patch 由 FA commit 直接生成且 base --check 过+同源码主树判据全绿（传递链完整）。
  - **★密度轴正式轮 r1 收账（density2_gap{16,24,32,48}，GPU3+私有 ext 缓存+vo）**：gap16=663.7+**8-hash 锚 8/8 自证**；gap24=666.1（counts {15,15}）；gap32=664.9（counts {15,12}=interval 密度 −25%）；gap48=660.8+**reqs 900/sentence 24=编队拆解相变在新 HEAD 复现**（窗0 定谳自洽）。**gap 24→32 数据裁决=速度中性**（16/24/32 三点 ±1.2 tps=<±1tps 纪律带内；拍板单"速度收益待正式轮"→正式轮判零收益）→**诚实收针=不改生产默认**；若为触发密度/精度预算考虑可另议（数据在案）。生产 gap 改值≠换锚件自证（判速 bench 恒 gap16，gap16 腿锚逐位不动）。**tps 多发参考带（本窗，非独占）=663-667**（D4/E/F/密度四发同窗；机器级真独占带=遗留，同机双 session 常态下不可达成）。**撞窗案回应（host 场 §终账 r1）：09:00 窗换 .so 崩其判据车属实=我方执行 P-2 .so 换装；已确认其隔离环境处置；接受新纪律=发车/换 .so 前互查对端窗口。**
  - **发布仓组装进行中**：源码 delta 32 文件已 staged（patches 全量+persistent_batch 删/codec 搬家镜像+benchmarks 两件+utils/hybrid_selectors/triton_kernel/sitecustomize）；tmp/=gitignore（ext prebuilt 非发布物，义务收窄=kernel_patches 源码级）；fa3 patch 域已核（26 文件零 cute）；**fa4 patch 预演失配**（P-2 碰 vfai→fa4 的 4 hunks 需 rebase：B-2 hunk 主树已含等价可剔/FA4-A1 guard 主树无需保留）；push 待 P-2 定局+patch 链重验。
- **host 场三线终账（10:30 级，全部停靠未合 main）**：
  - **P1-B ✓ 停靠 `e077c70` @p1b-gap-throttle**（[GAP-THROTTLE-GENERATIONAL] 方向 B 直译+施工中发现的下游三传票复审点必要项；sentence/lease 形班车保持全额 gap=避 A1 同型崩链收窄决策；interplay 19/19（T8 新语义重写+T8b/T8c 新增）+周边双态逐位同+CPU 复现 gap48 班车 nreq 3→7=方向 B 预期值；换锚窗欠账四项在 commit 正文）。
  - **L2 ✓ 停靠 `27cdb87` @l2-topk-tie**（两段法主案落 selector_pipeline_ext.py 两臂咽喉；12/12 单测+C++ standalone GPU 30/30 逐位==oracle；**知情项：ATen 组合税 5.2-9.4×（绝对 70-570µs/selector-refresh，e2e 预估 <0.5%），若 e2e 成材料优先定制 2-pass kernel，勿回退 stable-sort（否决域；但其 7.2GB 前提=全 K 排序、窄化域不落——留复核）；★C++ 改动=落地时 .so 需重打，与 P-2 协调成套**；换锚窗欠账=选集对拍/换锚三件重立/e2e 税判读）。
  - **Lite v1.1 ✓ 停靠 `6fae919` @lite-v11-arm-body（基座 c56ed44）**：移植全绿=正确性面全立（CPU 双态 394 失败集逐条同+passed 恰+16；GPU 判速×4 发 8/8 MATCH 现锚+counts 精确+黄金逐位+provenance 全查）。**★零命中定谳（诚实账）：现 HEAD 世代簇≈5 步连续 FULL（AMNESTY 后形态）击穿臂 v1「二步簇」假设——兑现中段步一次性消费稳态快照，真返程步 pre_trigger_snapshot_missing=永 refuse；臂=休眠臂（fail-close 全程正确=判据绿的机理）**。r1 撞窗案=对端 P-2 换 .so 窗（08:59-09:08）ABI 58→59 崩我方判据车，处置=隔离 FA 环境（session scratchpad fa_b08420e+04cbbf97 颗）单变量重跑=资产保留可复用；**双 session 同机判据新纪律=发车前互查对端是否在换 .so/重编窗**。
- **★host 场第二批认领（13:15，用户令"按顺序自主推进"）**：①ext 重编窗批（§7-D 全清单+刀A/D/E 凑批，弹药=scratchpad/ext_batch/ 九 patch，注意 #16 已退役双 ext 影子=逐 patch 对现实核对）②#14 ultra 零分配束刀 2-1..2-9 凑批③**独占 tps 正式带（机器全空窗口，现 HEAD 0aba147+成套 fc8dbbcf，官方口径×4 发）**④换锚窗（P1-B+L2 判据）随后。kernel 场若返场先看本行避让。
- **★第三批扩容认领（13:30，用户令"全都推进可并行不偷懒"）**：⑤CHUNK18 两均分 AB（新方向 #1，GPU6 半天）⑥#17 P1 pack 族迁 CUDA（128µs C++ 窗实证）⑦小件清扫批（fast-path 双次尝试缓存/U1/U2/topk pool snapshot/老队列三件翻案定谳/seq_lens 冻结文档）⑧tps 批后追 K-a/K-c 兑现对照腿（独占窗）。未认领余量=profile 15 通道合并+三巨型函数拆分（下一波补位）+fa4 patch rebase（kernel 维护，若对端不返场我方下波接）;LongBench/selected_k 维持"精度不测"拍板搁置;Lite PBD 臂+分支合入=待用户拍板。
- **★ext 重编窗批终报（停靠 `0e1925f` @ext-rebuild-batch=0aba147+2 commit，15 文件 +641/−170，判据全绿）**：九 patch=8 applied+1 弃用（姊妹已被 #16 退役）+连带补修 python shim 7→8 列漏网件；**★A1 长 ctx 雷根修升级案B=LSE 双核扫描域泛化（static_range 128 桶帽→runtime cdiv 循环，任意 K 精确+编译膨胀根因移除；K=131072×四窗口×双臂对拍 2/2 绿+负证=基线 kernel 同测 FAIL 确证测试能抓原雷）**；凑批诚实账=刀E memo −7.2µs/call（−37%）+刀A 噪声带+刀D 弃用（fused 入口=ATen 回退+丢语义+新姊妹漂移面）=净 ~1ms/run<刀单期望带，**价值主体=正确性硬化（A1+12 合同守卫）**；判据=8-hash 现锚 8/8+counts/612+黄金 2/2+kernel 24/24。**发布仓义务追加 4 ext prebuilt .so 重打**；**★合账知情项=主树脏区 14 文件（对端预 apply p1-p9，与本分支逐字节前缀重叠）→合入以本分支为准后 checkout 清**；新坑=pgrep 转义须覆盖命令文本内全部裸进程名（脚本文件执行免疫）+C++ 合同收紧须 grep python shim 调用面。
- **★#11 CHUNK18 AB 定谳（停靠 `1cd927f` @chunk18-ab=战报+判读脚本；零代码=纯 env AB×6 腿）**：**硬证据=refresh 世代成本 A 8374 vs B 7532µs=−10.1%（三对逐对 B<A 带无重叠；payload_build −11.4%+deferred_enqueue −12.5%；串行 commit 3→2=主收益=TP8 世代膨胀律兑现面）**；tps +0.45%（带内不作硬门）；**黄金 0.6b 两档位（14/14 与 18/10 不均分）均逐位对锚=数值面干净，4B 判速 hash 变（6/8）钉死为触发时序面（sentence 15→17）非红案**；三鸟=①key 收缩部分兑现（0.6b 反例:形状族 1→2=population/ring 收益须按新形态重验）②commit 3→2 兑现 −842.6µs/世代③arena 仅 +50.3MB 无害。**改默认 14→18=拍板件（数据面支持；代价=判速三锚必换（B 形态新锚组 run×3 已自证可立）+触发密度轻扰须复核交互面+无精度背书）——建议与换锚窗（P1-B+L2）并窗一次换锚省翻锚成本**。
- **★#17 P1 pack 迁 CUDA 收针（停靠 `45bcba5` @pack-cuda-p1，判据全绿）**：真身澄清=pack 族是 Triton JIT kernel（非纯 py），迁移刀=Triton launcher→**C++ ext 单发射**（新独立 ext `utils/req_meta_pack_ext.py`，三 kernel 逐语句直译=纯整数逐位等价；@triton.jit 保留为对拍 oracle 非双路径，先例同款+文件头合同）；消费面零改动。**收益=发射 med 27.6→16.1µs（−42%）、p99 56.6→16.6µs（−71%=Triton launcher 尾抖消除），每触发步 ~11µs**。判据：对拍 18/18 逐位+poison 哨兵、pack 合同 115/115、collect +18 恰等、8-hash 现锚 8/8+counts 精确+黄金逐位；**顺手根修 §7 留账基线红 test_req_meta_pack_sink_validation（39e0e33 遗留）转绿**。发布仓义务=纯 py+JIT ext 无需重打 FA .so。**★新坑入档：多 agent 并行窗 git stash 栈=仓库级共享有竞态（pop 冲突+栈混他人条目+tracked 改动一度丢失）——并行窗双态对照禁用 stash，先 commit 再 diff/checkout 对照；绝不触碰栈内他人条目**。
- **★小件清扫批收针（六件全清，停靠 `86f3ed6` @small-sweep=仅件6 文档；取证仪器全撤零残留）**：件1 fast-path 双次尝试缓存=**弃案**（实锤 56/342 步双跑但安全可跳子集仅 ~1.5ms/跑=0.03%<噪声地板，昂贵档与 update_classification 变异纠缠；前置筛选纪律+翻锚风险，安全设计留 commit 信息可复议）；件2 U1=**良性定谳无需修**（42 次全为 batch 组成变化=合法 FULL_RECOMPILE）；件3 U2=落点集中 deadline-drain 61%+writer-flush 22% 按世代节拍=对 epoch 门升强制**有利**，升格前需 U5 配套；件4 topk pool 0.6GB UNKNOWN=**消掉**（实为默认池 capture 缓冲：capture_scores 侧输出 536.9MB+step layout 176.2MB，非 graph mempool=原标签误判；**显存削减靶点=capture_scores，记账**）；件5 老队列三件翻案定谳=serve drain**已过时**（judge v3 replay-aware 根修，7a1869a 起健康）/installer B **主体已完成**（唯一残余=**B-3 REPLAY_WRAPPER_ABLATE ~90-120 行纯死码可整删,记入下一波清淤批**；B-4/B-5 有消费者缓做）/bundle 64k **已被三旋钮转正吸收**（ef12af2）；件6 `SFI_SEQ_LENS_CARRIER_FREEZE_2026-07-12.md` 七载体冻结契约入仓（当前 HEAD 逐点复核）。
- **★tps 正式带立带（13:31 收针，拍板单④+遗留件收账）**：现 HEAD 0aba147+成套 fc8dbbcf 四发（tpsband_r1-r4，GPU3 同卡交替）=**662.5/663.9/663.7/666.7，8-hash 8/8 MATCH×4（正典口径重算）+counts {sentence:15,interval:16}/612 精确×4+provenance 全对**——与 kernel 场不同窗四发（663-667）完全重合=**tps 正式带定为 663-667（跨 session 8 发互证）**。诚实注记：本窗 5 agent CPU 底噪（load 3-10）非绝对纯净，但跨窗形态一致=带稳定自证；**⑧K-a/K-c 单独归因腿=诚实收针不再追**（两刀已合 main、收益已被带吸收，多 agent 常态下纯净归因窗无行动价值）。
  - **★Lite 适配刀终账（12:00 级，分支四 commit 链完整交付=`6fae919`→`f38860c`（N 步窗语义）→`a5c8740`（截存缺口根修：v1 唯一截存窗被两形态绕过→三点共用捕获，capture 32=每世代全覆盖）→`cb9178e`（消费点终位：兑现中段步 collect 面假稳态实锤→消费移至 extrapolate 成功=返程承接确认点+authority 真值门；终态））**。判据全绿：CPU 20+20 合同+interplay 17/17+stash 双态失败集逐条同×3 轮；判速 8-hash 现锚 **8/8 MATCH×8 发**+counts 精确×8+黄金逐位×5（provenance 对齐 cb9178e）。**★结构性收益定谳：命中 6/跑（armX 同源短簇族）+snapshot_missing 48→10；38 个长簇返程步=页界/recent 窗尽头相位的设计内诚实 miss（必须走 PBD 慢路径，extrapolate 拒=正确行为）；tps 七发 665.9-670.1 与合入前无可辨差=+0.8-1.2% 预期在现 HEAD 不成立（876ffa1 时代账：慢路径已被 K-a/K-c/L1 等多刀瘦身，臂边际收益缩水至噪声下）**。J3 仪器五键+refuse trace 随树=下窗判读资产。**呈拍板：①分支处置（推荐=停靠保留，墙钟纪律不建议现状合入零收益 1500 行大件）②PBD 形态臂立项与否（38 诚实 miss 转命中=剩余收益面，extrapolate 放行页滑+plan 整建重铺 first/count=手册 §12-1 自然延伸；新刀独立判据）**。

## §12.3（07-12 下午，kernel 场 session）ext 批九连+14 红根修+triton 死件清理=main `e234a33` 落地

- **★main 0aba147→`e234a33`（16 文件，+406/−1163）+发布仓 `fc5b0d1`（320e5fb 之上 10 文件；★已 push=`320e5fb..fc5b0d1`→LV-NUS/SFI cuda-kernel，用户授权 07-12 下午"授权处理"）**。FA .so 零动（fc8dbbcf 成套不变）；selector ext 家族源变=拉取方首跑 JIT 重编数分钟（发布仓 commit message+远端指引增补段已注记，含 env gate 严格 '0'/'1' 新合同提示）。
- **①ext 九 patch 全落**（弹药=scratchpad/ext_batch/patches p1-p9，九连 CLEAN）：p1 高危#2 meta 合同 ≥8→≥10；p2 softnms/crosshead 守卫；p3 pagesize；p4 卫生三件（warp-align 姊妹/env 严格 0|1/bounds C++ 合同+launch check）；p5 rrp_bind arraylen；p6 host reduce 单源；p7 A1/A2 scan domain（A1=长 ctx 真雷）；p8 A3 i64；p9 F2 stride_pad fail-fast+F3 pack 输入合同。
- **②★14 红根修（本窗最重定谳，勿重查）**：test_selector_log_s_ext 的 log_f_pre 族 14 红=**p1 CHECK≥10 杀死 wrapper legacy 7 列桥（size(1)==7 升格 col1←last_n*pad/col7←pad）→fixture 扩 10 列绕桥直塞→col1 语义分叉（桥语义=pad 行宽 vs kernel v2 语义=scratch head stride=last_n*capacity，生产真值 postprocess.py:582）→CUDA 臂 head≥1 全错位读**。修=fixture 按 v2 契约重填+**legacy 7 列桥整删**（恒 10 列下必炸死代码，7 列零活调用，违无 fallback 铁律）。单变量排除链在案：p9 where/col7 值/行寻址/p4 env/新 .so 全无罪。教训=**"7 列时代绿"的机理是 wrapper 桥不是 OOB 巧合**——列消费定谳必须把 python wrapper 层与 C++ 层一起看。
- **③triton 参考臂退役**（用户拍板 triton 全面下线=清理不维护）：14 测参考臂 triton lastn_gt1→**同 ext scalar 版**（独立 kernel；批式独有判别面=meta 解码/stride 派生/head 写域完整对拍，token 级共享数学由黄金链兜底）；测试改名 matches_scalar_reference。
- **④triton 死件删单落地**（agent 符号级全仓盘点）：cross_head 族+soft_nms 族 −868 行+3 死测试（test_cross_head_mutex_smooth 整删+bounds 删 soft_nms 对拍），零残留。**留活件**：attn_reference/rows1 fuse+log_f LSE 族（生产 CUDA ext 唯一数值 oracle）/flash_attn_score_dump_fwd.py 全文件（pack 三件生产活+lastn_gt1 oracle）。**triton 下线余量路线图**：pack 三件=host 场 #17 已停靠 `45bcba5`（C++ ext 单发射，triton 保留为对拍 oracle=同款处置哲学）；oracle 组终退役=torch 纯参考重写后（fused_log_f_prior 族 4 GPU 测+key_norms 第 4 测+log_f_kernel_alpha 同批），列下窗。**冲突面提示：e234a33 动了 flash_attn_score_dump_fwd.py（p9）与 45bcba5 的 pack 域相邻，该分支 rebase 时注意**。
- **判据账**：CPU ring_war 11 绿+collect 3089（=HEAD−3 净删对账）；GPU 定向 11 文件 218 passed+**20 预存红（HEAD 裸 stash 腿同 20 红=铁预存：test_resolved_row_ptr_arena 10+test_mixed_page_full_cudagraph_replay_hook 10，mixed_page/rrp 合同欠账非本批；另 key_norms arena_expansion 1 红同预存）**；黄金 bs2 {636fb032,44e80946} 逐位+counts {s:2,i:2}/84（**补发腿=gate_passed+production_gate 双 True/reasons 全空=完整 gate 绿形态在案，golden_extbatch2**）；判速 vo=8-hash 正典锚 8/8 逐位+counts {15,16}/612+tps 666.0（带内；同机并行口径）+route=ResolvedRowPtr。
- **新坑三条**：①黄金 one-shot 发车正典必带 `--producer-mode full-open-gt1`（缺省 legacy→child 踩 stale runner boundary 守卫 SystemExit 2=route missing 形态假红）+`--outputs-include-text`（缺→semantic gate missing_text 噪声，token_ids hash 判据不受影响）；②Monitor/pgrep 自匹配（监视命令文本含关键词→pgrep 匹配监视器自身=死监视永不退出；[x] 转义纪律扩展到 Monitor）；③预检 && 链与 nohup 发车必须分条（`&& ... &` 整链后台化=预检输出丢失）。
- **#14 撞车收案**：我方 agent 施工 launch_template 视图化+200×12 A/B 逐位 PASS 后发现 host 场 `u14-zero-alloc` 已 commit 同刀（面更大 2-1+2-9+2-8）→按 skip 纪律撤回工作树改动等其分支；A/B harness 留 scratchpad=独立收敛互证。CHUNK18 同款：host 场 `1cd927f` 已定谳，我方 agent 零 GPU 独立核验通过（黄金 hash 原始重算对锚）。
- **★取证批 B 收账（四件全交代）+irregular 案终谳=main `3a0ff14`**：①topk pool snapshot/②U1/③G6=并行场已收案（86f3ed6 件4/件2+0615654，我方 agent 逐一独立核证属实，勿重取证）；**④irregular +1.49%（bf17ebe J1 遗留"另案"）正式收案**：影子对（batch_row vs irregular）按构造非 iso-locality（rowptr 构造路径也不同）→补 `kind4_rowptr_irregular_stride1_non_tma` 控制双胞胎（同构造路径唯 stride 3→1，页集=连续前缀=selected oracle 默认参考逐位适用）。**正典档定谳：batch_row +1.094%/irregular +2.402%/stride1 +0.875%（vs_raw_native）→ stride1≈batch_row=构造路径结构税≈0；irregular−stride1=+1.53% 单变量=页集局部性钉死（与原案 +1.49% 吻合）**；三 case same_split 逐位 0.0+gate True。旁证=kv36864/sel256 稀选集档三 case 零差（页集效应随形态消失）+该档 `oracle_split_status=split_mismatch_fallback`=非正典档 pin 不可复现的设计内诚实曝光（勿误读为红）。**速度门判读纪律：irregular case 的 +2.4% 级=已知构造性页集混杂，结构税以 stride1 控制腿为准**。
- **★★host 场合入窗开启（14:4x，排他声明：窗内 kernel 场请勿推 main，预计 ≤1h）**：合入序=pack-cuda-p1(45bcba5)→small-sweep(86f3ed6)→chunk18-ab(1cd927f 文档)→u14-zero-alloc(等 agent 终报)→ext 增量提取（对端 e234a33 已落九 patch，我方 0e1925f 多出四件=A1 案B 扫描域泛化替换案A raise+python shim 7→8 列修+刀E memo −37%+刀A 去重——原 agent 增量移植中）；四支与 e234a33 文件零交集+merge 预演零冲突已验。合完总闸一发（判速 hash+counts+黄金）。随后换锚窗（P1-B+L2 rebase+换锚腿，候选新锚呈拍板不合 main）。
- **▲kernel 场应答（时间线交叉自查+rebase 冲突面清单，写于看到上条后）**：上条排他声明写入时我方两笔已在飞行中落地=`a703cc9`（TRITON-FINAL-RETIRE）+`ccbf85e`（21 预存红追平）——**声明写入晚于我方最后一次读档，commit 前未重查=我方协作缺口，致歉；此后 main 冻结直到贵场窗关**（后续文档笔/memory 均不推 main）。**给贵场合入序的冲突面自查清单**：①`flash_attn_score_dump_fwd.py`：我删 lastn_gt1 族（原 728-955，含 p9 F2 hunk）+头注去引用；**pack 族一行未动**（45bcba5 的对拍 oracle 依赖面完好，130 项消费者测试自证）——45bcba5 rebase 若有本文件 hunk 注意行号漂移；②`test_req_meta_pack_sink_validation.py`：我改第三对锚（换文件尾 next_fn=None）+迁入 2 个 sink 负向合同（自 test_log_f_kernel_alpha 退役件）——**与 45bcba5 的"顺手根修此文件基线红"大概率同文件冲突，rebase 时以两边语义并集为准**（我方改动=锚修+迁测，无行为断言变化）；③**`alpha_selector_kernel.py` 已整删**→贵场 ext 增量四件中「A1 案B 扫描域泛化」宿主已消灭=**可直接弃件**（雷随宿主删除，torch 参考替身=tests/reference_log_f_prior.py 以 C++ 为正典）；「python shim 7→8 列修」的宿主（wrapper legacy 7 列桥）也已在 e234a33 整删（恒 10 列合同）=**同弃件**；刀E memo/刀A 去重若作用于 selector ext C++ 与我方九 patch 叠加,增量提取时以 e234a33 后的文件现状为基线重取 diff；④`ccbf85e` 动 rrp_arena/mixed_page_hook/key_norms 三测试（纯期望追平,实现零动）——与 u14-zero-alloc（metadata_builder/launch_template 域）零交集。贵场总闸一发时我方两笔的判据背书：黄金 gate 全绿（golden_extbatch2）+判速 vo 8/8+合并测试腿 185 passed+生产 import 面零变化（a703cc9/ccbf85e 均纯测试+死件）。

---

# §13(07-12 午后)host 场收工终账+host 开发分支 merge 指引

## §13.1 本窗全部落地(main 链,host 开发分支从最新 main 拉出即得全部)

授权窗+推进窗全账(时序):`c56ed44`(拍板批落档)→对端 `3606c95`(oracle 同 s)/`0aba147`(P-2 gitlink)→`e234a33`(对端 ext 九 patch)→`3a0ff14`(irregular 终谳)→`e0dc356`(交接档滚动账)→**`3823c50`(#17-P1 pack 迁 CUDA)→`b7c6465`(小件批件6)→`a2a9cec`(#11 CHUNK18 定谳)**→对端 `a703cc9`(triton 终下线 −2201)/`ccbf85e`(21 预存红追平)→**`7458eef`(ext 增量终局:刀E/刀A/writer;A1 案B 随 triton 整删弃=雷已消灭)→`32b1015`(#14 零分配+U3 结案)**。**★组合形态总闸全绿(32b1015 上):CPU 冒烟 39 passed(pack 对拍+ext 替身版)+判速 8-hash 现锚 8/8 逐位+counts {sentence:15,interval:16}/612 精确+tps 665.6(663-667 带内)+provenance 自证+黄金 bs2 双 child {636fb032,44e80946} 逐位 MATCH+gate True——本窗全部合入内容判据链闭合,32b1015=host 开发分支基线**。(黄金腿一次假红勘误:one-shot 必带 `--output` 必选参数,缺则 argparse rc=2,勿误读为门红。)

## §13.2 host 开发分支 merge 指引(★接续开发者从此节进入)

**基线**:直接从最新 main 拉 host 开发分支——本窗全部已合内容(上表)零额外动作即得。**成套铁律=FA vendored `4c405dc`+`.so fc8dbbcf8eaa`**(动 FA 必三件套成套重打;主树 build/ 缓存有 stale 前科,重编须从新源自证)。

**停靠分支处置表(merge 决策全在此)**:

| 分支 | commit | 性质 | merge 指引 |
|---|---|---|---|
| `p1b-gap-throttle` | e077c70 | ★换锚件 | **绝不单独 merge**。已在换锚窗树预合+判据(见 §13.3);新锚用户拍板转正后,由 `git merge p1b-gap-throttle && git merge l2-topk-tie` 双支同窗合 main+锚值文档同步换,再跑一发黄金+判速自证 |
| `l2-topk-tie` | 27cdb87 | ★换锚件 | 同上,与 P1-B 绑定同窗;其 selector_pipeline_ext.py 与已合刀A/刀E 同文件不同区(自动合安全,合后 C++ 重编由判据车自动覆盖) |
| `lite-v11-arm-body` | cb9178e(四 commit 链) | 待拍板(非换锚) | 拍"合入"→直接 merge(判据全绿在案:8/8×8 发+黄金×5;merge 后一发判速 hash 对现锚+黄金复验即可);拍"封存"→留档勿删(J3 仪器+PBD 臂基座);拍"PBD 臂立项"→在此分支继续施工 |
| `lite-v11-pinned-guard` | e3a281a | 已收编 | 内容已进 main(9c561c1 R1'),**可删** |
| `ext-rebuild-batch`/`ext-delta-rebase` | 0e1925f/2db71ca | 已收编 | 真增量已由 7458eef 终局合入,分支留证据链,阅后**可删** |
| `pack-cuda-p1`/`small-sweep`/`chunk18-ab`/`u14-zero-alloc` | 45bcba5/86f3ed6/1cd927f/6d02463 | 已合 | **可删** |
| `anchor-window` | e8f41ae(=main+P1-B+L2) | 换锚窗验证树 | 拍板后按上行处置;worktree 在 `.claude/worktrees/anchor-window` |

**merge 纪律(host 接续开发必守)**:①换锚件(触发节奏/选集成员改动)绝不单独合,凑窗一次换锚(判速 8-hash+counts+tps 带三锚重立+黄金复验,新锚呈用户拍板后才转正);②行为改动 merge 后必跑黄金 bs2(one-shot `--preset bs2long-cap128 --producer-mode full-open-gt1 --outputs-include-text`,★勿直跑 run_sparse_only/勿缺 producer-mode=假红)+判速 bs8x12k hash;③多 session/多 agent 并行窗:**禁 stash**(仓库级共享有竞态,先 commit 再 diff 对照)+私有 `TORCH_EXTENSIONS_DIR`(ext JIT 乒乓)+发车/换 .so 前互查对端窗口+pgrep [x] 转义覆盖命令文本内全部裸进程名;④worktree e2e 三件套=`VLLM_SPARSE_FA3_UPSTREAM_ROOT` 指主树 FA+corpus 从主树 out/ 拷+私有 ext 缓存。

**当前判据锚(最终 release `a8456f5`)**:8-hash 正典 `{fff0445e, c0a6376d, 99a9aea2, 7db6f96d, fda56972, deef29a8, 8e8e2dca, 8778b158}`+counts `{sentence:19, interval:16}`/payloads 756+黄金 `{ad585b37, 44e80946}`+本地 A100 all-decode tps **654.5-656.9**。旧锚依赖 step-0 被误判为未初始化的 min-gap 绕过 bug，仅保留作单变量复现证据；最终因果见 §13.6。

## §13.3 换锚窗结果(已收针)

**★换锚窗收针=判据面零翻锚,P1-B+L2 已合 main(`754c9c3`/`e2f5252`)**——判速 ×3(anchorw_r1-r3,GPU0):8-hash **现锚 8/8 逐位 ×3+run 间互同**,counts {sentence:15,interval:16}/612 精确 ×3,tps 662.0-663.7,provenance=e8f41ae;黄金 bs2(FA symlink 修通 worktree 探针坑后):**{636fb032,44e80946} 双 child 逐位+gate True**。机理=判速 bench gap16 下 P1-B 降档带未改变 commit 集合、L2 护栏在当前档位选集本就正典——**行为差只在非判据形态(gap≥24 生产/gap48 相变/大尺度并列)=两支设计意图**。⇒ 拍板单"新锚转正"项自动消解(三锚全保持);§13.2 表中 P1-B/L2 两行过时(已合,以本节为准);anchor-window 分支/worktree 已收编可清。合入序全账与新坑三条(worktree 黄金腿 sitecustomize 探针 FA symlink 处方/bench 静默 return 2 分诊=json stderr_tail/--output 必选)=`SFI_MERGE_GUIDE_2026-07-12.md` §0.5(该手册由 kernel 场起稿+host 场状态刷新,**后续 agent 合流唯一入口**)。

## §13.4 拍板件汇总(全部呈用户,收工报告同步)

1. **换锚窗新锚转正**(§13.3 值)——拍则 P1-B+L2 合 main+锚值文档换。
2. **CHUNK18 默认 14→18**(世代成本 −10.1% 硬证据;换锚件,若拍可并 §13.3 同窗重立锚省一次翻锚)。
3. **Lite 分支处置**(推荐停靠保留)+**PBD 形态臂立项**(38 诚实 miss 转命中=剩余收益面)。
4. L2 深化(e2e 税若成材料→定制 2-pass kernel;stable-sort 否决前提复核)。

## §13.5 未动余量(接续开发候选,按价值序)

profile 15 通道合并(−800~1200 行,等本窗全支合完后做=冲突面已清)/B-3 REPLAY_WRAPPER_ABLATE 死码整删(~100 行)/三巨型函数拆分(重构类,判据成本高)/fa4 patch rebase(SM100,对端注记)/PBD 形态臂(拍板件③)/epoch 门升强制(U2 数据有利,待 U5 配套)/capture_scores 537MB 显存靶点(记账)。维持搁置:LongBench/selected_k(用户拍"精度不测")/FA kernel 深水+带宽封路区(停止线)。

## §13.6 最终 release 合仓与旧锚反转（07-12 夜，覆盖 §13.3 的 release 判据）

**release 生产提交=`a8456f5`，P1-B/L2/既有 kernel 仓位已对齐落地。** 主树与
release 的 7 个生产文件逐字节一致；Lite v1.1/CHUNK18 未合。FA3 多轮 kernel
优化已经包含在 release 基座 `3325c0d`，本增量未改 FA gitlink/patch，因此已编
该基座的机器无需二次编 FA3；全新或更老基座仍必须编。L2 的
`selector_pipeline_ext` 必须重建，语义版本=`2026071201`，旧 prebuilt 会被拒绝。

最终审计补了四类真实缺陷：① selector stale `.so` 静默加载；②合法
`last_decode_refresh_step=0` 被 `or -1` 吞掉；③三处 persistent pinned-host
source 在 non-blocking H2D 未完成时复用；④ layer-zero deadline 与
merge-policy-off mixed bus 的排列不变量。定向验证 82/82，selector C++/GPU
tie oracle 30/30，kernel-only gate=True，collect=3120。

§13.3 的“零翻锚”只描述修复 step-0 前的历史树，不能继续作为 release 判据。
严格单变量 A/B：最终代码黄金=`{ad585b37,44e80946}`、producer work
`[16,111,112]`；仅把 step-0 helper 回退为旧 `or -1`，立即逐位恢复
`{636fb032,44e80946}` 与 `[4,99,100]`。这证明旧锚依赖 min-gap 绕过 bug。
用户已批准新锚转正；bs8x12k 三发 8-hash 逐位互同、counts/payloads 精确同，
值以本节顶部“当前判据锚”为准。远端操作入口=
`REMOTE_TEST_INSTRUCTIONS_2026-07-12.md`。

# §14（07-12 夜续窗）本地 L2/runner/显存审计优化收针（晚于 §13.6）

> 详细证据与接续顺序=`SFI_LOCAL_OPTIMIZATION_CLOSEOUT_2026-07-12.md`。本节是
> release `a8456f5` 之上的本地候选工作树，尚未产生新 release commit；因此
> §13.6 仍是当前已发布状态，§14 是下一同步批。

## §14.1 本地已落地

1. **L2 定制 CUDA**：旧 ATen post-topk 组合换为稳定 CUDA 压紧，semantic=
   `2026071203`，module/cache=`selector_pipeline_ext_v2026071203`。TP8 代理总耗时
   11.038→6.071 ms（1.818x），post-topk 7.239→1.847 ms（3.92x），峰值临时
   显存 766.31→37.45 MiB（−95.1%），checksum 同；相关 selector 87 passed，
   runner+capture+selector 整合定向 89 passed。
2. **stale dlopen/重复 JIT 根修**：同名 `.so` 原地重编后当前进程仍可能持有旧
   映像，故把 semantic 写入扩展名；JIT 返回后再校验 semantic/entrypoint。
   selector 同步固定 `TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES=1`，消除 import
   顺序交替改 Ninja command 导致的每腿重复重编。
3. **runner fail closed**：不再 `|| true` 吞 child rc；prewarm/postflight
   selector path/hash/actual+expected semantic 与 `run_provenance` 必须一致；每
   TAG 非阻塞 flock（无 flock rc69、TAG 冲突 rc75）。VERDICT_ONLY allowlist 精确
   只有 `semantic_output_health_not_ok:unknown_without_reference`、
   `interval_trigger_intents_below_expected`；旧档把
   `async_producer_writer_missing` 当 vo 噪声的口径作废，它是真红。
   default18 VO 终复验已闭环：route writer-complete count=36 与 full
   refresh-profile count=36 精确一致；provenance 从 `sparse_constants` 单真源读到
   18，producer reasons 空、`SPEED RUN OK`。route count 是 measurement，绝非
   allowlist；writer missing 继续真红。
4. **capture scratch 探针**：默认关闭，启用时按 allocation key 在 JSON/锁 I/O
   前去重。full 4B/bs8x12k 实测 `[2,256,32,2,16384]` fp16=1 GiB：benchmark
   child 漏传 `max_num_seqs`，vLLM 默认 256，是真实 over-allocation。命令链 pin
   `max_num_seqs=batch_size=8` 的代码根修已落，目标 `[2,8,32,2,16384]`=
   32 MiB（32x/−96.875%）；针对性 12 passed、py_compile/diff-check 绿。根修后
   full 4B c14/c18 的 prebuild/live 均实测 32 MiB，live cache_hit=true，闭环。
5. **kernel-only vacuous gate 根修**：旧 broad `--case-filter non_tma` 可能令
   production RRP 对全 skip、aggregate 空证据误绿；现已 fail closed。production
   pinned 三 case + `--force-num-splits 1` 实测 gate=true、max RRP
   overhead=0.215745%≤0.4%。broad split1 三次 direct_physical=0.428263% 红但不是
   production RRP，诚实记录、不阻断 selector merge。多 cell 进一步按
   `(batch,requested_kv_len,case)` 聚合 missing/duplicate/launch/speed evidence，逐
   record 检查 RRP，防止一格 matched 掩盖另一格 skipped；合同 25 passed。

## §14.2 停止线与默认决策

- **PBD no-go**：唯一可外推 miss=8；现实可回收 12.189 ms/2.9848 s=0.408%，
  全清零绝对上界 0.736%，均低于 1% gate，不投约 1500 行 arm。§13.4/§13.5
  的 PBD“待立项”状态由本节关闭。
- **CHUNK18 promotion 为全局默认**：4B 六腿 chunk14 均值 658.0226 TPS，
  chunk18 664.7054 TPS（+1.0156%，paired 三组全正）。golden c14/c18 均
  gate/prod=True、`{ad585b37,44e80946}`、counts 2/2、payload 84；full c14=
  662.2989 TPS 且命中旧 8-hash/19+16/756，full c18=670.0263 TPS 且命中新锚
  `{970c061e,c0a6376d,99a9aea2,03eb3325,fda56972,deef29a8,f6d008c1,4076e3d5}`/
  16+16/648。证据链已按“语义变更”闭合，`sparse_constants.py` 默认正式
  14→18，并有 promotion 注释/默认测试；env=14 保留回滚。
- **FA3 不重编**：本续窗没有改 FA gitlink/FA3 源/patch；`3325c0d` 已编机器只需
  selector 1203 一次重建。全新或旧于 3325c0d 的机器仍按正常流程编 FA3。

## §14.3 下一接力

`max_num_seqs=batch_size` 根修、full 4B 32 MiB/cached-hit、golden c14/c18、非 VO full
c14/c18 与 production RRP pinned 负控均已完成（禁止退回 vacuous broad），且
selector 前后 hash 稳定；default18 provenance+VO writer-complete 也已闭环。
最终合同并集复跑 **436 passed**；default18 导致的 chunk/ready 固定值测试债、
post-bridge min-gap 双边与 writer-default provenance 单真源已同步收口；262K
arena 预算也已改为跟随 chunk/in-flight 单真源（default18=5 GiB），live i32
seq-lens 在预备阶段落位并保持 refresh cache-hit。
接续只需把生产文件逐字节同步
release 工作树、补 release commit hash 与远端指令。**commit/push 必须另取用户
明确确认**；不要把本节“本地已落地”误读成“已发布”。

## §14.4 2026-07-13 最终接力（覆盖 §14.3）

本地 kernel/host 合仓位和后续 infra 优化均已落到两棵工作树：29 个生产文件
逐字节一致，尚未 commit/push。selector semantic 已从 1203 升为
`2026071301`，L2/L1 融合后生产形状 `1.977344→1.901568 ms`，10/10 逐位等价；
发布树 `.so` SHA256=
`995421642e5ff51ab2cf59d66b96a6f54c595d10dda162c658043387c9de694e`。远端必须
重建 selector；FA3 源/gitlink/patch 没变，已有正确 FA3 基座无需重编。

host/infra 同步收口了四类隐藏成本与错误边界：prefill 36 层重复 lookup/bind/
lease 降至 2/2/2；forward capture absent mapping 分配 36→1 并移除强 truth
路径 216 个 device assert kernel；authority 缺失的 StepBound 永不 memo；runner
对 ABI cache、GPU/KV/FA3、精确 corpus、nonce/artifact 与完整 token 输出全部
fail closed。Qwen 8×12k corpus cache hit `0.06 s`，实际 token 长度 8/8 均 12000。

最终 A100/Qwen3-4B bs8×12k 实车：sparse
`268.9904 decode TPS / 665.8281 all-decode TPS`，dense
`175.9740 / 292.7292`，提升 `1.5286x / 2.2746x`；两车均 8×256 完整输出，
route/FA3/provenance 绿。统一 release gate `617 passed, 3 skipped`，红队 29 项
clean。Lite v1.1 仍是未合停靠件，其专属合同不进入本次 gate。接续动作仅剩：
核对最终 diff 后，向用户取得 `git commit`/`git push` 明确确认；未确认前禁止
提交或发布。
