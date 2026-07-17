#!/usr/bin/env python
"""[SPARSE-LIVENESS-JUDGE 2026-07-13 v4] serve/LongBench 场 sparse 活性判官。

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
  R1 世代活性: refresh flush 记录中 refresh_payloads>0 的世代发布次数，或
     replay-batched 路径的正 refresh-payload-enqueue route 事件，二者之一
     ≥ --min-world-publish。后者是当前 runtime 明确不写 per-chunk profile
     时的同源提交证据，不能把健康 replay 误判为 decode 恒 dense。
  R2 触发活性: sentence 触发意图>0；interval-only 配置由 R1 的世代活性
     覆盖，replay-batched 路径也读取 route 事件中的 refresh_reason。
  R3 路由活性(两源,mmap 优先):
     a) --route-counter-mmap: 每个 TP rank 独占 10×int64 槽
        (total/kind4/has_rrp/kind0..4/compact_steps/compact_rows)，聚合前
        先要求各 rank 完全一致；kind4>0=前向真走 sparse resolver；
     b) --route-trace: 逐事件 JSON 解析 mode 分布+row_is_compact 聚合
        (长请求 sparse 化率;短请求低于阈值恒 dense 是设计内,聚合占比
        才有判读价值)。
  R4 请求级 fallback: --step-trace 提供 fa3_step_state 时按 PID/request
     重建相位。prefill、bootstrap 未完成的 dense 行与显式 refresh（包括
     bootstrap refresh）允许 dense/native；已进入 compact 的请求若回到
     mature decode dense 则判红。已知长请求可用
     --require-mature-decode-compact 收紧为所有 mature decode 都必须 compact。
  配置组合防护(启动期,非本判官):dual-gen×residency 缺失/FORCE_DENSE
     冲突=安装事务 fail-fast;residency slots<max_num_seqs=profile
     dummy_run 预检 fail-fast([SERVE-LIVENESS-PREFLIGHT])。本判官管
     "跑完之后的活性证据",预检管"起服务之前的配置死刑"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from collections import Counter


def _iter_profile_records(path: str, *, offset: int = 0):
    with open(path, "r", errors="ignore") as fh:
        fh.seek(offset)
        for line in fh:
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            try:
                yield parts[0], parts[1], json.loads(parts[2])
            except json.JSONDecodeError:
                continue


def _iter_json_records(
    path: str,
    *,
    offset: int = 0,
    parse_errors: list[str] | None = None,
):
    with open(path, "r", errors="ignore") as fh:
        fh.seek(offset)
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if parse_errors is not None:
                    parse_errors.append(
                        f"offset 后第 {line_number} 行 JSON 损坏: {exc.msg}"
                    )
                continue
            if isinstance(value, dict):
                yield value
            elif parse_errors is not None:
                parse_errors.append(
                    f"offset 后第 {line_number} 行不是 JSON object"
                )


def _read_route_counter_mmap(path: str):
    """Aggregate fixed 10×int64 single-writer slots without hiding rank drift."""
    with open(path, "rb") as fh:
        raw = fh.read()
    slot_bytes = 10 * 8
    if len(raw) < slot_bytes or len(raw) % slot_bytes != 0:
        return None
    slots = [
        tuple(
            int(value)
            for value in struct.unpack_from("10q", raw, rank * slot_bytes)
        )
        for rank in range(len(raw) // slot_bytes)
    ]
    values = tuple(sum(slot[index] for slot in slots) for index in range(10))
    return {
        "total": values[0],
        "kind4": values[1],
        "has_resolved_row_ptr": values[2],
        "by_kind": dict(enumerate(values[3:8])),
        "compact_row_steps": values[8],
        "compact_rows": values[9],
        "rank_slots": len(slots),
        "rank_consistent": all(slot == slots[0] for slot in slots[1:]),
        "per_rank": [list(slot) for slot in slots],
    }


def _read_replay_batched_refresh_evidence(
    path: str, *, offset: int = 0
) -> dict[str, int]:
    """Read fresh generation evidence emitted before replay-batched deferral.

    The normal profile path records ``refresh_payloads`` at this same enqueue
    boundary.  Replay-batched flush intentionally omits per-chunk profile
    records, so this route event is its equivalent producer-commit evidence.
    """
    publishes = 0
    payloads = 0
    sentence_intents = 0
    malformed = 0
    parse_errors: list[str] = []
    for event in _iter_json_records(
        path,
        offset=offset,
        parse_errors=parse_errors,
    ):
        if event.get("event") != (
            "mixed_page_full_cudagraph_replay_refresh_payload_enqueue"
        ):
            continue
        payload_count = event.get("payload_count")
        if type(payload_count) is not int or payload_count <= 0:
            malformed += 1
            continue
        publishes += 1
        payloads += payload_count
        if str(event.get("refresh_reason", "")) == "sentence":
            sentence_intents += 1
    return {
        "publishes": publishes,
        "payloads": payloads,
        "sentence_intents": sentence_intents,
        "malformed": malformed,
        "parse_errors": len(parse_errors),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-profile-log", required=True)
    ap.add_argument("--route-trace", default="")
    ap.add_argument("--route-counter-mmap", default="")
    ap.add_argument("--step-trace", default="")
    ap.add_argument("--min-world-publish", type=int, default=1)
    ap.add_argument(
        "--refresh-profile-offset",
        type=int,
        default=0,
        help="只判定该字节偏移之后的新 refresh 记录",
    )
    ap.add_argument(
        "--route-trace-offset",
        type=int,
        default=0,
        help="只判定该字节偏移之后的新 route 记录",
    )
    ap.add_argument(
        "--baseline-compact-row-steps",
        type=int,
        default=-1,
        help="R3c 基线计数;非负时要求本轮 compact 读步数有增量",
    )
    ap.add_argument(
        "--min-compact-row-step-delta",
        type=int,
        default=0,
        help="相对 R3c 基线所需的最小 compact 读步数增量",
    )
    ap.add_argument(
        "--run-since",
        type=float,
        default=0.0,
        help="run 起点 epoch 秒;产物 mtime 早于它=残留文件判红",
    )
    ap.add_argument(
        "--step-trace-offset",
        type=int,
        default=0,
        help="只判定该字节偏移之后的新 request-level step 记录",
    )
    ap.add_argument(
        "--reject-request-fallback",
        action="store_true",
        help="请求进入 compact 后若回到 mature decode dense/native 则判红",
    )
    ap.add_argument(
        "--require-mature-decode-compact",
        action="store_true",
        help="已知长请求中所有非 refresh 的 mature decode 行都必须 compact",
    )
    ap.add_argument("--phase", default="run", help="输出中使用的评测相位标签")
    args = ap.parse_args()

    fails = []
    warns = []
    profile_offset = max(0, args.refresh_profile_offset)
    if args.refresh_profile_offset < 0:
        fails.append("R0: --refresh-profile-offset 必须非负")
    route_trace_offset = max(0, args.route_trace_offset)
    if args.route_trace_offset < 0:
        fails.append("R0: --route-trace-offset 必须非负")
    step_trace_offset = max(0, args.step_trace_offset)
    if args.step_trace_offset < 0:
        fails.append("R0: --step-trace-offset 必须非负")
    if args.min_compact_row_step_delta < 0:
        fails.append("R3c: --min-compact-row-step-delta 必须非负")
    if (
        args.min_compact_row_step_delta > 0
        and args.baseline_compact_row_steps < 0
    ):
        fails.append(
            "R3c: 要求 compact 增量时必须提供非负 "
            "--baseline-compact-row-steps"
        )

    # ---- R0 产物存在/新鲜(残留 append 假 PASS 与"patch 未装"分开点名) ----
    def _r0_check(
        path: str,
        label: str,
        required: bool,
        *,
        minimum_size: int = 0,
    ) -> bool:
        if not path:
            return False
        if not os.path.isfile(path):
            msg = (
                f"R0: {label} 不存在({path})—— env 未设 / sitecustomize 未生效"
                "(PYTHONPATH 缺仓根?)/ patch 未安装。这不是零活性,是零证据。"
            )
            (fails if required else warns).append(msg)
            return False
        size = os.path.getsize(path)
        if size <= minimum_size:
            if minimum_size > 0:
                msg = (
                    f"R0: {label} 没有偏移 {minimum_size} 之后的新记录"
                    f"({path}, size={size})"
                )
            else:
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

    profile_ok = _r0_check(
        args.refresh_profile_log,
        "refresh profile log",
        True,
        minimum_size=profile_offset,
    )
    trace_ok = _r0_check(
        args.route_trace,
        "route trace",
        False,
        minimum_size=route_trace_offset,
    )
    mmap_ok = _r0_check(args.route_counter_mmap, "route counter mmap", False)
    require_step_trace = bool(
        args.reject_request_fallback or args.require_mature_decode_compact
    )
    step_trace_ok = _r0_check(
        args.step_trace,
        "request-level step trace",
        require_step_trace,
        minimum_size=step_trace_offset,
    )

    # ---- R1/R2 世代与触发活性 ----
    # ``refresh_payloads`` is the physical profile source.  On the documented
    # replay-batched path no per-chunk profile record exists, while the same
    # logical producer generation emits a positive route enqueue event.
    profile_world_publishes = 0
    profile_payloads_total = 0
    reasons = Counter()
    pids = set()
    if profile_ok:
        for pid, tag, rec in _iter_profile_records(
            args.refresh_profile_log,
            offset=profile_offset,
        ):
            pids.add(pid)
            rp = int(rec.get("refresh_payloads", 0) or 0)
            if rp > 0:
                profile_world_publishes += 1
                profile_payloads_total += rp
            reasons["sentence"] += int(rec.get("sentence_trigger_intents", 0) or 0)

    replay_evidence = {
        "publishes": 0,
        "payloads": 0,
        "sentence_intents": 0,
        "malformed": 0,
        "parse_errors": 0,
    }
    if trace_ok:
        replay_evidence = _read_replay_batched_refresh_evidence(
            args.route_trace,
            offset=route_trace_offset,
        )

    # Profile remains authoritative whenever it observed a physical publish.
    # Otherwise the replay enqueue is the only observable logical generation;
    # do not double-count a generation visible through both sources.
    if profile_world_publishes > 0:
        generation_count = profile_world_publishes
        payloads_total = profile_payloads_total
        generation_source = "refresh_profile"
    elif replay_evidence["publishes"] > 0:
        generation_count = replay_evidence["publishes"]
        payloads_total = replay_evidence["payloads"]
        generation_source = "replay_batched_route"
    else:
        generation_count = 0
        payloads_total = 0
        generation_source = "none"
    sentence_intents = max(
        int(reasons.get("sentence", 0)),
        int(replay_evidence["sentence_intents"]),
    )

    if profile_ok:
        print(
            f"R1 世代活性: generation_count={generation_count} "
            f"payloads_total={payloads_total} pids={len(pids)} "
            f"source={generation_source} profile_world_publish="
            f"{profile_world_publishes} replay_enqueue_events="
            f"{replay_evidence['publishes']} replay_payloads_total="
            f"{replay_evidence['payloads']}"
        )
        if (
            generation_source == "replay_batched_route"
            and (
                replay_evidence["malformed"] > 0
                or replay_evidence["parse_errors"] > 0
            )
        ):
            fails.append(
                "R1: replay route 证据不完整: "
                f"invalid_payload_count={replay_evidence['malformed']} "
                f"json_parse_errors={replay_evidence['parse_errors']}"
            )
        if generation_count < args.min_world_publish:
            fails.append(
                f"R1: logical refresh 世代 {generation_count} < "
                f"{args.min_world_publish} —— 本相位没有新的 selector refresh "
                "generation；sparse 读侧是否工作由 R3c/R4 独立判定"
            )
        if sentence_intents <= 0 and generation_count <= 0:
            fails.append("R2: 触发意图为零(sentence=0 且无逻辑世代)")
        else:
            print(f"R2 触发活性: sentence_intents={sentence_intents}")

    # ---- R3a 路由活性(mmap 结构化计数,主判据) ----
    compact_row_steps = -1
    route_rank_slots = 0
    if mmap_ok:
        counters = _read_route_counter_mmap(args.route_counter_mmap)
        if counters is None:
            fails.append(
                "R3a: route counter mmap 不是非空的 80B-per-rank 完整槽"
                "(损坏/旧 64B 合同/未初始化)"
            )
        else:
            route_rank_slots = int(counters["rank_slots"])
            total = counters["total"]
            kind4 = counters["kind4"]
            compact_row_steps = int(counters.get("compact_row_steps", -1))
            ratio = (kind4 / total) if total else 0.0
            print(
                f"R3a 路由活性(mmap): total={total} kind4={kind4} "
                f"({ratio:.1%}) has_rrp={counters['has_resolved_row_ptr']} "
                f"by_kind={counters['by_kind']} rank_slots={counters['rank_slots']}"
            )
            if not counters["rank_consistent"]:
                fails.append(
                    "R3a: TP rank-local route/liveness counters diverged; "
                    f"per_rank={counters['per_rank']}"
                )
            if total > 0 and kind4 == 0:
                fails.append(
                    "R3a: mmap 计数 kind4=0 —— 前向从未走 sparse resolver"
                )
            # ---- R3c 读侧活性(step build 侧计数,replay-aware 主判据) ----
            # [JUDGE-REPLAY-AWARE 2026-07-09] FULL-graph serve 下 python 侧
            # 路由计数/trace 只在 capture/eager/prefill 步发射,replay 步不可
            # 见——旧 R3b 在健康引擎上恒 FAIL(eager 定谳:18576/18684 步
            # compact 健康,dense 事件全是 bootstrap/prefill 窗)。本判据由
            # step_context_worker 每步 bump,graph 无关,是 replay 覆盖的读侧
            # 真值。旧版 8q/单共享槽产物已在 R3a fail closed。
            if compact_row_steps >= 0:
                print(
                    f"R3c 读侧活性(step 计数): compact_row_steps="
                    f"{compact_row_steps} compact_rows={counters['compact_rows']}"
                )
                if compact_row_steps == 0 and generation_count > 0:
                    fails.append(
                        "R3c: 世代已发布但没有任何 step 存在 compact 读行 "
                        "——写而不读(读侧装配断点)"
                    )
                if args.baseline_compact_row_steps >= 0:
                    compact_delta = (
                        compact_row_steps - args.baseline_compact_row_steps
                    )
                    print(
                        "R3c 本轮增量: "
                        f"baseline={args.baseline_compact_row_steps} "
                        f"delta={compact_delta}"
                    )
                    if compact_delta < args.min_compact_row_step_delta:
                        fails.append(
                            "R3c: 本轮 compact 读步数增量 "
                            f"{compact_delta} < "
                            f"{args.min_compact_row_step_delta}"
                        )

    if args.baseline_compact_row_steps >= 0 and not mmap_ok:
        fails.append("R3c: 已请求本轮增量判定，但 route counter mmap 不可用")
    elif args.baseline_compact_row_steps >= 0 and compact_row_steps < 0:
        fails.append("R3c: route counter mmap 不含 replay-aware compact 读计数")

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
        # [JUDGE-REPLAY-AWARE 2026-07-09] trace 的 row_is_compact 只覆盖
        # python 可见步(capture/eager/prefill)——FULL-graph 下 replay 步不发
        # 事件,"全 False"在健康引擎上是常态(那些步 dense 本合法)。降档:
        # 仅当 R3c(step 计数)不可用(旧运行时)时才以此判死;R3c 可用时给
        # 信息行,以 R3c 为准。
        if total_rows > 0 and compact_rows == 0:
            if compact_row_steps >= 0:
                print(
                    "[INFO] R3b: trace 可见步(capture/eager/prefill 窗)全 "
                    "dense=FULL-graph 常态,读侧真值以 R3c 为准"
                )
            else:
                fails.append(
                    "R3b: row_is_compact 全 False 且无 R3c 计数(旧运行时)"
                    "——所有可见行 dense(请求全短于阈值?FORCE_* env?"
                    "bootstrap 未完成即结束?)"
                )
    if not (mmap_ok or trace_ok):
        warns.append("R3 未判(--route-trace / --route-counter-mmap 均不可用)")

    # ---- R4 请求级 fallback：只把 mature decode 的不期望 dense 判死。 ----
    # prefill / bootstrap / refresh 是显式相位，不属于 fallback；短请求可以一直
    # dense。full eval 用 compact_seen 后的 sticky 规则，已知长 smoke 额外要求
    # 每个 mature decode 行从一开始就 compact。
    if step_trace_ok:
        compact_seen: set[tuple[int, str]] = set()
        step_events = 0
        traced_rows = 0
        compact_rows = 0
        prefill_dense_rows = 0
        bootstrap_dense_rows = 0
        refresh_rows = 0
        precompact_dense_rows = 0
        fallback_rows: list[str] = []
        step_trace_parse_errors: list[str] = []
        tp_step_pids: set[int] = set()
        tp_step_groups: dict[int, tuple[str, int, set[int]]] = {}
        tp_parity_mismatches: list[str] = []
        tp_duplicate_events: list[str] = []
        tp_missing_fields: list[str] = []
        tp_parity_mismatch_count = 0
        tp_duplicate_event_count = 0
        tp_missing_field_count = 0
        vector_fields = (
            "req_ids",
            "is_prefill_by_row",
            "bootstrap_done_by_row",
            "use_compact_by_row",
            "row_mode_by_row",
            "layer_effective_refresh_by_row",
        )
        tp_parity_fields = (
            "epoch",
            "step_handle_id",
            "step_handle_generation",
            "step_identity_token",
            "batch_size",
            "rows_traced",
            "req_ids",
            "is_prefill_by_row",
            "bootstrap_done_by_row",
            "use_compact_by_row",
            "dispatch_logf_producer_by_row",
            "logits_last_n_by_row",
            "row_mode_by_row",
            "layer_effective_refresh_by_row",
        )
        for event in _iter_json_records(
            args.step_trace,
            offset=step_trace_offset,
            parse_errors=step_trace_parse_errors,
        ):
            if event.get("event") != "fa3_step_state":
                continue
            step_events += 1
            try:
                raw_pid = event["pid"]
                raw_batch_size = event["batch_size"]
                raw_rows = event["rows_traced"]
                vectors = {field: event[field] for field in vector_fields}
            except KeyError:
                fails.append(
                    "R4: fa3_step_state 缺少 PID/batch_size/rows_traced/行向量"
                )
                continue
            if any(
                type(value) is not int
                for value in (raw_pid, raw_batch_size, raw_rows)
            ):
                fails.append(
                    "R4: fa3_step_state PID/batch_size/rows_traced 必须是整数"
                )
                continue
            pid = raw_pid
            batch_size = raw_batch_size
            rows = raw_rows
            if batch_size <= 0 or rows != batch_size or any(
                not isinstance(values, list) or len(values) != rows
                for values in vectors.values()
            ):
                fails.append(
                    "R4: fa3_step_state 必须完整覆盖 batch_size 且行向量等长"
                )
                continue
            tp_step_pids.add(pid)
            missing_parity = tuple(
                field for field in tp_parity_fields if field not in event
            )
            if missing_parity:
                tp_missing_field_count += 1
                if len(tp_missing_fields) < 8:
                    tp_missing_fields.append(
                        f"pid={pid} missing={missing_parity}"
                    )
            else:
                identity_token = event["step_identity_token"]
                if type(identity_token) is not int:
                    tp_missing_field_count += 1
                    if len(tp_missing_fields) < 8:
                        tp_missing_fields.append(
                            f"pid={pid} step_identity_token 非整数"
                        )
                else:
                    parity_payload = {
                        field: event[field] for field in tp_parity_fields
                    }
                    digest = hashlib.sha256(
                        json.dumps(
                            parity_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ).encode("utf-8")
                    ).hexdigest()
                    group = tp_step_groups.get(identity_token)
                    if group is None:
                        tp_step_groups[identity_token] = (digest, pid, {pid})
                    else:
                        reference_digest, reference_pid, seen_pids = group
                        if pid in seen_pids:
                            tp_duplicate_event_count += 1
                            if len(tp_duplicate_events) < 8:
                                tp_duplicate_events.append(
                                    f"token={identity_token} pid={pid}"
                                )
                        else:
                            seen_pids.add(pid)
                        if digest != reference_digest:
                            tp_parity_mismatch_count += 1
                            if len(tp_parity_mismatches) < 8:
                                tp_parity_mismatches.append(
                                    f"token={identity_token} "
                                    f"reference_pid={reference_pid} pid={pid}"
                                )
            traced_rows += rows
            for row in range(rows):
                req_id = vectors["req_ids"][row]
                boolean_values = (
                    vectors["is_prefill_by_row"][row],
                    vectors["bootstrap_done_by_row"][row],
                    vectors["use_compact_by_row"][row],
                    vectors["layer_effective_refresh_by_row"][row],
                )
                row_mode = vectors["row_mode_by_row"][row]
                if not isinstance(req_id, str) or not req_id:
                    fails.append(
                        "R4: request-level step trace 的 req_id 必须是非空字符串"
                    )
                    continue
                if any(type(value) is not bool for value in boolean_values):
                    fails.append(
                        f"R4: req={req_id!r} 相位/use_compact/refresh 行向量必须是 bool"
                    )
                    continue
                if type(row_mode) is not int:
                    fails.append("R4: row_mode_by_row 含非整数值")
                    continue
                is_prefill, bootstrap_done, use_compact, layer_refresh = (
                    boolean_values
                )
                if use_compact != (row_mode == 1):
                    fails.append(
                        f"R4: req={req_id!r} use_compact 与 row_mode={row_mode} 不一致"
                    )
                    continue
                key = (pid, req_id)
                if is_prefill:
                    # request_id is caller-owned and may be reused after a
                    # request finishes. A new prefill is the explicit
                    # lifecycle boundary carried by this trace, so discard
                    # sticky state from the previous incarnation first.
                    compact_seen.discard(key)
                    if row_mode not in (0, 2) or layer_refresh:
                        fails.append(
                            f"R4: req={req_id!r} prefill 相位非法 "
                            f"row_mode={row_mode} refresh={layer_refresh}"
                        )
                        continue
                    prefill_dense_rows += 1
                    continue
                # Keep the judge's phase precedence identical to
                # resolve_decode_row_policy(): an explicit decode refresh is
                # legal before bootstrap_done because that refresh is what
                # materializes the first compact generation.
                if row_mode == 3:
                    if not layer_refresh:
                        fails.append(
                            f"R4: req={req_id!r} row_mode=3 缺少显式 refresh 标记"
                        )
                        continue
                    refresh_rows += 1
                    continue
                if layer_refresh:
                    fails.append(
                        f"R4: req={req_id!r} 显式 refresh 行未使用 row_mode=3"
                    )
                    continue
                if not bootstrap_done:
                    if row_mode != 0:
                        fails.append(
                            f"R4: req={req_id!r} bootstrap 相位非法 "
                            f"row_mode={row_mode} refresh={layer_refresh}"
                        )
                        continue
                    bootstrap_dense_rows += 1
                    continue
                if row_mode == 1:
                    compact_seen.add(key)
                    compact_rows += 1
                    continue
                if row_mode != 0:
                    fails.append(
                        f"R4: req={req_id!r} mature decode 使用未知 row_mode={row_mode}"
                    )
                    continue

                strict_failure = bool(args.require_mature_decode_compact)
                sticky_failure = bool(
                    args.reject_request_fallback and key in compact_seen
                )
                if strict_failure or sticky_failure:
                    fallback_rows.append(f"pid={pid},req={req_id}")
                else:
                    precompact_dense_rows += 1

        if step_trace_parse_errors:
            sample = "; ".join(step_trace_parse_errors[:4])
            fails.append(
                "R4: request-level step trace 含损坏记录: "
                f"count={len(step_trace_parse_errors)} sample=[{sample}]"
            )

        print(
            f"R4 请求级相位({args.phase}): events={step_events} rows={traced_rows} "
            f"compact={compact_rows} prefill_dense={prefill_dense_rows} "
            f"bootstrap_dense={bootstrap_dense_rows} refresh={refresh_rows} "
            f"precompact_or_short_dense={precompact_dense_rows} "
            f"fallback={len(fallback_rows)}"
        )
        if require_step_trace and step_events == 0:
            fails.append("R4: 没有 fa3_step_state 请求级记录")
        if args.require_mature_decode_compact and compact_rows == 0:
            fails.append("R4: 已知长请求没有任何 mature compact decode 行")
        if fallback_rows:
            sample = ", ".join(fallback_rows[:8])
            fails.append(
                "R4: 检出非预期 mature decode dense/native fallback: "
                f"count={len(fallback_rows)} sample=[{sample}]"
            )

        # ---- R5 TP step semantics: aggregate counters can prove a final
        # mismatch but cannot identify a transient compact/native split.  The
        # step identity is scheduler-derived and must carry byte-identical row
        # policy on every rank.  Only a possibly split first/last JSONL group
        # is tolerated because offset snapshots and a live writer may cut at a
        # rank boundary; any interior incompleteness is terminal.
        expected_tp_ranks = (
            route_rank_slots if route_rank_slots > 0 else len(tp_step_pids)
        )
        if expected_tp_ranks > 1:
            incomplete_tokens: list[str] = []
            incomplete_token_count = 0
            if tp_step_groups:
                boundary_tokens = {
                    min(tp_step_groups),
                    max(tp_step_groups),
                }
                for token, (_digest, _reference_pid, seen_pids) in (
                    tp_step_groups.items()
                ):
                    if len(seen_pids) != expected_tp_ranks and token not in boundary_tokens:
                        incomplete_token_count += 1
                        if len(incomplete_tokens) < 8:
                            incomplete_tokens.append(
                                f"token={token} pids={sorted(seen_pids)}"
                            )
            complete_groups = sum(
                1
                for _digest, _reference_pid, seen_pids in tp_step_groups.values()
                if len(seen_pids) == expected_tp_ranks
            )
            print(
                "R5 TP step 语义一致性: "
                f"expected_ranks={expected_tp_ranks} "
                f"observed_pids={sorted(tp_step_pids)} "
                f"groups={len(tp_step_groups)} complete={complete_groups} "
                f"mismatch={tp_parity_mismatch_count} "
                f"interior_incomplete={incomplete_token_count}"
            )
            if len(tp_step_pids) != expected_tp_ranks:
                fails.append(
                    "R5: step trace 的 TP PID 数与 route mmap 不一致: "
                    f"expected={expected_tp_ranks} observed={sorted(tp_step_pids)}"
                )
            if tp_missing_field_count:
                fails.append(
                    "R5: TP step trace 缺少一致性字段: "
                    f"count={tp_missing_field_count} sample={tp_missing_fields}"
                )
            if tp_duplicate_event_count:
                fails.append(
                    "R5: 同一 TP rank 重复写入 step identity: "
                    f"count={tp_duplicate_event_count} sample={tp_duplicate_events}"
                )
            if tp_parity_mismatch_count:
                fails.append(
                    "R5: TP rank 的逐 step row-policy 语义分叉: "
                    f"count={tp_parity_mismatch_count} sample={tp_parity_mismatches}"
                )
            if incomplete_token_count:
                fails.append(
                    "R5: TP step trace 中段缺 rank 事件: "
                    f"count={incomplete_token_count} sample={incomplete_tokens}"
                )
            if not tp_step_groups:
                fails.append("R5: TP>1 但没有可比较的 step identity")
        elif step_events > 0:
            print(
                "R5 TP step 语义一致性: single-rank run, "
                f"observed_pids={sorted(tp_step_pids)}"
            )
    elif require_step_trace:
        fails.append("R4: 请求了 fallback 判定但 request-level step trace 不可用")

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
