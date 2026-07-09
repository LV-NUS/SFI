#!/usr/bin/env python
"""[SPARSE-LIVENESS-JUDGE 2026-07-09 v2] serve/LongBench 场 sparse 活性判官。

bench 场有 producer gate 兜活性;serve 注入场(LongBench 等)此前裸奔——
sparse 若静默退化成 dense(触发为零/patch 未装/路由回落),精度数字看着
正常但测的其实是 dense。本判官对 serve 产物开箱判定,不依赖 bench harness。

用法(serve 侧起服务时带 env,产物即判据源;路径带时间戳防残留假 PASS):
  RUN_TS=$(date +%s)
  VLLM_SPARSE_REFRESH_PROFILE=1 \
  VLLM_SPARSE_REFRESH_PROFILE_LOG=/tmp/lb_refresh_profile.${RUN_TS}.log \
  VLLM_SPARSE_FA3_ROUTE_TRACE_LOG=/tmp/lb_route.${RUN_TS}.jsonl \
  VLLM_SPARSE_FA3_ROUTE_COUNTER_MMAP=/tmp/lb_route_counter.${RUN_TS}.bin \
  <启动 serve;跑完 LongBench 后:>
  python scripts/check_sparse_liveness.py \
      --refresh-profile-log /tmp/lb_refresh_profile.${RUN_TS}.log \
      --route-trace /tmp/lb_route.${RUN_TS}.jsonl \
      --route-counter-mmap /tmp/lb_route_counter.${RUN_TS}.bin \
      --run-since ${RUN_TS} [--min-world-publish 1]

判定(全过=exit 0,任一红=exit 1 并逐条点名):
  R0 产物新鲜/存在: profile log 必须存在且非空(缺失/空=patch 未装或
     env 未设,与"装了但零活性"是不同的病,分开点名);--run-since 给出
     run 起点 epoch 时,产物 mtime 早于它=残留文件(append 模式的假 PASS
     陷阱),判红。
  R1 世代活性: refresh flush 记录中 refresh_payloads>0 的世代发布次数
     ≥ --min-world-publish(=0 即从未 sparse 化,decode 恒 dense)。
  R2 触发活性: sentence 触发意图>0(interval 触发不在本记录,靠 R1 免
     误报:interval-only 配置下 R1>0 即活)。
  R3 路由活性(两源,mmap 优先):
     a) --route-counter-mmap: 8×int64(total/kind4/has_rrp/kind0..4),
        kind4>0 且占比给出=前向真走 sparse resolver(结构化计数,零
        字符串 grep);
     b) --route-trace: 逐事件 JSON 解析 mode 分布+row_is_compact 聚合
        (长请求 sparse 化率;短请求低于阈值恒 dense 是设计内,聚合占比
        才有判读价值)。
  配置组合防护(启动期,非本判官):dual-gen×residency 缺失/FORCE_DENSE
     冲突=安装事务 fail-fast;residency slots<max_num_seqs=profile
     dummy_run 预检 fail-fast([SERVE-LIVENESS-PREFLIGHT])。本判官管
     "跑完之后的活性证据",预检管"起服务之前的配置死刑"。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import Counter


def _iter_profile_records(path: str):
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            try:
                yield parts[0], parts[1], json.loads(parts[2])
            except json.JSONDecodeError:
                continue


def _read_route_counter_mmap(path: str):
    """8×int64: total / kind4 / has_resolved_row_ptr / kind0..kind4."""
    with open(path, "rb") as fh:
        raw = fh.read(64)
    if len(raw) < 64:
        return None
    values = struct.unpack("8q", raw)
    return {
        "total": values[0],
        "kind4": values[1],
        "has_resolved_row_ptr": values[2],
        "by_kind": dict(enumerate(values[3:8])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-profile-log", required=True)
    ap.add_argument("--route-trace", default="")
    ap.add_argument("--route-counter-mmap", default="")
    ap.add_argument("--min-world-publish", type=int, default=1)
    ap.add_argument(
        "--run-since",
        type=float,
        default=0.0,
        help="run 起点 epoch 秒;产物 mtime 早于它=残留文件判红",
    )
    args = ap.parse_args()

    fails = []
    warns = []

    # ---- R0 产物存在/新鲜(残留 append 假 PASS 与"patch 未装"分开点名) ----
    def _r0_check(path: str, label: str, required: bool) -> bool:
        if not path:
            return False
        if not os.path.isfile(path):
            msg = (
                f"R0: {label} 不存在({path})—— env 未设 / sitecustomize 未生效"
                "(PYTHONPATH 缺仓根?)/ patch 未安装。这不是零活性,是零证据。"
            )
            (fails if required else warns).append(msg)
            return False
        if os.path.getsize(path) == 0:
            msg = f"R0: {label} 为空({path})—— 注入生效但从未产出记录"
            (fails if required else warns).append(msg)
            return False
        if args.run_since > 0 and os.path.getmtime(path) < args.run_since:
            fails.append(
                f"R0: {label} mtime 早于 --run-since(残留产物;append 模式下"
                "旧 run 记录会假 PASS)。给产物路径带时间戳重跑。"
            )
            return False
        return True

    profile_ok = _r0_check(args.refresh_profile_log, "refresh profile log", True)
    trace_ok = _r0_check(args.route_trace, "route trace", False)
    mmap_ok = _r0_check(args.route_counter_mmap, "route counter mmap", False)

    # ---- R1/R2 世代与触发活性 ----
    world_publishes = 0
    payloads_total = 0
    reasons = Counter()
    pids = set()
    if profile_ok:
        for pid, tag, rec in _iter_profile_records(args.refresh_profile_log):
            pids.add(pid)
            rp = int(rec.get("refresh_payloads", 0) or 0)
            if rp > 0:
                world_publishes += 1
                payloads_total += rp
            reasons["sentence"] += int(rec.get("sentence_trigger_intents", 0) or 0)
        print(
            f"R1 世代活性: world_publish={world_publishes} "
            f"payloads_total={payloads_total} pids={len(pids)}"
        )
        if world_publishes < args.min_world_publish:
            fails.append(
                f"R1: refresh 世代发布 {world_publishes} < {args.min_world_publish}"
                " —— decode 从未 sparse 化(疑触发哑火/请求全短于阈值)"
            )
        if reasons.get("sentence", 0) <= 0 and world_publishes <= 0:
            fails.append("R2: 触发意图为零(sentence=0 且无世代发布)")
        else:
            print(f"R2 触发活性: sentence_intents={reasons.get('sentence', 0)}")

    # ---- R3a 路由活性(mmap 结构化计数,主判据) ----
    if mmap_ok:
        counters = _read_route_counter_mmap(args.route_counter_mmap)
        if counters is None:
            fails.append("R3a: route counter mmap 小于 64B(损坏/未初始化)")
        else:
            total = counters["total"]
            kind4 = counters["kind4"]
            ratio = (kind4 / total) if total else 0.0
            print(
                f"R3a 路由活性(mmap): total={total} kind4={kind4} "
                f"({ratio:.1%}) has_rrp={counters['has_resolved_row_ptr']} "
                f"by_kind={counters['by_kind']}"
            )
            if total > 0 and kind4 == 0:
                fails.append(
                    "R3a: mmap 计数 kind4=0 —— 前向从未走 sparse resolver"
                )

    # ---- R3b 路由活性(trace 结构化解析;字符串 grep 已废:旧匹配串
    #      '"dense_native"' 全仓无源=恒 0 假安静) ----
    if trace_ok:
        mode_counts = Counter()
        compact_rows = 0
        total_rows = 0
        events = 0
        with open(args.route_trace, "r", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events += 1
                mode = str(evt.get("mode", evt.get("probe", "?")))
                mode_counts[mode] += 1
                rows = evt.get("row_is_compact")
                if isinstance(rows, (list, tuple)):
                    total_rows += len(rows)
                    compact_rows += sum(1 for v in rows if v)
        rrp = mode_counts.get("resolved_row_ptr", 0)
        row_ratio = (compact_rows / total_rows) if total_rows else 0.0
        print(
            f"R3b 路由活性(trace): events={events} modes={dict(mode_counts)} "
            f"compact_rows={compact_rows}/{total_rows} ({row_ratio:.1%})"
        )
        if events > 0 and rrp == 0:
            fails.append(
                "R3b: route trace 无 resolved_row_ptr 事件=前向从未走 sparse"
            )
        if total_rows > 0 and compact_rows == 0:
            fails.append(
                "R3b: row_is_compact 全 False —— 所有行 dense(请求全短于"
                "阈值?FORCE_* env?bootstrap 未完成即结束?)"
            )
    if not (mmap_ok or trace_ok):
        warns.append("R3 未判(--route-trace / --route-counter-mmap 均不可用)")

    warns.append(
        "配置组合(slots≥max_num_seqs / dual-gen×residency / FORCE_*)由启动期"
        "预检覆盖([SERVE-LIVENESS-PREFLIGHT]);serve 若起服务成功即已过检"
    )

    for w in warns:
        print(f"[WARN] {w}")
    if fails:
        for f in fails:
            print(f"[FAIL] {f}")
        print("SPARSE LIVENESS: FAIL")
        return 1
    print("SPARSE LIVENESS: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
