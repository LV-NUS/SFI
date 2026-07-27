#!/usr/bin/env python
"""[SPARSE-LIVENESS-JUDGE 2026-07-27 v8] serve/LongBench 场 sparse 活性判官。

bench 场有 producer gate 兜活性;serve 注入场(LongBench 等)此前裸奔——
sparse 若静默退化成 dense(触发为零/patch 未装/路由回落),精度数字看着
正常但测的其实是 dense。本判官对 serve 产物开箱判定,不依赖 bench harness。

用法(serve 侧起服务时带 env,产物即判据源;路径带时间戳防残留假 PASS):
  RUN_TS=$(date +%s)
  VLLM_SPARSE_FA3_ROUTE_TRACE_LOG=/tmp/lb_route.${RUN_TS}.jsonl \
  <启动 serve;跑完 LongBench 后:>
  python scripts/check_sparse_liveness.py \
      --route-trace /tmp/lb_route.${RUN_TS}.jsonl \
      --route-counter-snapshot /tmp/lb_route_counter.${RUN_TS}.bin \
      --run-since ${RUN_TS} [--min-world-publish 1]

判定(全过=exit 0,任一红=exit 1 并逐条点名):
  R0 产物新鲜/存在: FULL replay 的 route trace、冻结 counter 与按需 step
     trace 必须来自当前 fresh offset 窗；--run-since 给出 run 起点 epoch
     时，判据 mtime 早于它=残留文件，判红。
  R1 世代活性: TP-complete replay refresh-payload-enqueue world generation
     ≥ --min-world-publish。FULL replay 不产出 per-chunk refresh profile，
     旧 profile authority 已退休，缺 route enqueue 必须 fail-closed。
  R2 触发活性: sentence 触发意图>0；interval-only 配置由 R1 的世代活性
     覆盖，replay-batched 路径也读取 route 事件中的 refresh_reason。
  R3 路由活性(冻结 RPC + FULL replay hook 联合判定):
     a) --route-counter-snapshot: blocking worker RPC 后按 TP rank 冻结的
        10×int64 记录
        (total/kind4/has_rrp/kind0..4/compact_steps/compact_rows)，聚合前
        先要求各 rank 完全一致；kind4>0=Python 可见前向真走 resolver；
        FULL replay 下 kind4 不可见，改由 TP-complete graph hook 与
        replay-aware compact step counter 联合定谳；
     b) --route-trace: 仅在 fresh offset 窗内解析 TP-complete FULL replay
        graph hook。旧 direct-route/mode/逐行 compact 旁路权威已退休。
  R4 请求级 fallback: --step-trace 提供 fa3_step_state 时按 PID/request
     重建相位。prefill、row policy 未就绪的 dense 行与显式 refresh 允许
     dense/native；已进入 compact 的请求若回到 dense 或 policy 未就绪
     则判红。compact threshold 前的 mature dense 是设计内相位，不得误报。
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


def _iter_json_records(
    path: str,
    *,
    offset: int = 0,
    parse_errors: list[str] | None = None,
):
    # Offsets come from os.path.getsize(), so consume the trace as bytes.
    # Text-mode cookies and errors="ignore" can respectively mis-seek or hide
    # corruption, allowing stale/malformed evidence to enter a fresh window.
    with open(path, "rb") as fh:
        fh.seek(offset)
        for line_number, raw_line in enumerate(fh, start=1):
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                if parse_errors is not None:
                    parse_errors.append(
                        f"offset 后第 {line_number} 行 UTF-8 损坏: {exc.reason}"
                    )
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


def _read_route_counter_snapshot(path: str):
    """Aggregate a frozen 10×int64-per-rank RPC artifact without hiding drift."""
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
    path: str,
    *,
    offset: int = 0,
    expected_ranks: int = 0,
) -> dict[str, int]:
    """Read fresh generation evidence emitted before replay-batched deferral.

    Replay-batched flush intentionally has no per-chunk profile carrier.  The
    TP-complete enqueue group is the producer-commit authority for this mode.
    """
    groups: dict[tuple[int, str, str, tuple[str, ...]], dict[int, int]] = {}
    invalid_groups: set[tuple[int, str, str, tuple[str, ...]]] = set()
    enqueue_events = 0
    malformed = 0
    duplicates = 0
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
        enqueue_events += 1
        pid = event.get("pid")
        step_id = event.get("step_id")
        graph_key = event.get("graph_key")
        refresh_reason = event.get("refresh_reason")
        req_ids = event.get("refresh_intent_req_ids")
        payload_count = event.get("payload_count")
        if (
            type(pid) is not int
            or type(step_id) is not int
            or not isinstance(graph_key, str)
            or not graph_key
            or not isinstance(refresh_reason, str)
            or not refresh_reason
            or not isinstance(req_ids, list)
            or not req_ids
            or any(not isinstance(req_id, str) or not req_id for req_id in req_ids)
            or type(payload_count) is not int
            or payload_count <= 0
        ):
            malformed += 1
            continue
        group_key = (
            step_id,
            graph_key,
            refresh_reason,
            tuple(req_ids),
        )
        by_pid = groups.setdefault(group_key, {})
        if pid in by_pid:
            duplicates += 1
            invalid_groups.add(group_key)
            continue
        by_pid[pid] = payload_count

    publishes = 0
    payloads = 0
    sentence_intents = 0
    tp_incomplete = 0
    payload_mismatch = 0
    rank_contract_missing = int(bool(groups) and expected_ranks <= 0)
    for group_key, by_pid in groups.items():
        if group_key in invalid_groups:
            continue
        if expected_ranks <= 0 or len(by_pid) != expected_ranks:
            tp_incomplete += 1
            continue
        payload_counts = set(by_pid.values())
        if len(payload_counts) != 1:
            payload_mismatch += 1
            continue
        publishes += 1
        payloads += next(iter(payload_counts))
        if group_key[2] == "sentence":
            sentence_intents += 1
    return {
        "publishes": publishes,
        "payloads": payloads,
        "sentence_intents": sentence_intents,
        "enqueue_events": enqueue_events,
        "malformed": malformed,
        "duplicates": duplicates,
        "tp_incomplete": tp_incomplete,
        "payload_mismatch": payload_mismatch,
        "rank_contract_missing": rank_contract_missing,
        "parse_errors": len(parse_errors),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route-trace", default="")
    ap.add_argument(
        "--route-counter-snapshot",
        default="",
        help="blocking worker-RPC 后冻结的 10×int64-per-rank 二进制产物",
    )
    ap.add_argument("--step-trace", default="")
    ap.add_argument("--min-world-publish", type=int, default=1)
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
        "--expected-route-counter-ranks",
        type=int,
        default=0,
        help="非零时要求冻结快照恰好覆盖该数量的 TP ranks",
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
        help=(
            "请求进入 compact 后若回到 dense/native 或 policy-unready 则判红"
        ),
    )
    ap.add_argument(
        "--require-mixed-prefill-decode",
        action="store_true",
        help=(
            "要求至少一个 TP 完整一致的 step 同时包含 prefill/decode，"
            "且 row policy 同时包含 compact/native"
        ),
    )
    ap.add_argument("--phase", default="run", help="输出中使用的评测相位标签")
    args = ap.parse_args()

    fails = []
    warns = []
    route_trace_offset = max(0, args.route_trace_offset)
    if args.route_trace_offset < 0:
        fails.append("R0: --route-trace-offset 必须非负")
    step_trace_offset = max(0, args.step_trace_offset)
    if args.step_trace_offset < 0:
        fails.append("R0: --step-trace-offset 必须非负")
    if args.min_compact_row_step_delta < 0:
        fails.append("R3c: --min-compact-row-step-delta 必须非负")
    if args.expected_route_counter_ranks < 0:
        fails.append("R0: --expected-route-counter-ranks 必须非负")
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
            (fails if required else warns).append(
                f"R0: {label} mtime 早于 --run-since"
                "(残留产物;append 模式下旧 run 记录会假 PASS)。"
                "给产物路径带时间戳重跑。"
            )
            return False
        return True

    trace_ok = _r0_check(
        args.route_trace,
        "route trace",
        False,
        minimum_size=route_trace_offset,
    )
    snapshot_ok = _r0_check(
        args.route_counter_snapshot,
        "frozen route counter snapshot",
        False,
    )
    require_step_trace = bool(
        args.reject_request_fallback
        or args.require_mixed_prefill_decode
    )
    step_trace_ok = _r0_check(
        args.step_trace,
        "request-level step trace",
        require_step_trace,
        minimum_size=step_trace_offset,
    )
    snapshot_counters = (
        _read_route_counter_snapshot(args.route_counter_snapshot)
        if snapshot_ok
        else None
    )
    expected_replay_ranks = args.expected_route_counter_ranks
    if expected_replay_ranks <= 0 and snapshot_counters is not None:
        expected_replay_ranks = int(snapshot_counters["rank_slots"])

    # ---- R1/R2 FULL replay 世代与触发活性 ----
    replay_evidence = {
        "publishes": 0,
        "payloads": 0,
        "sentence_intents": 0,
        "enqueue_events": 0,
        "malformed": 0,
        "duplicates": 0,
        "tp_incomplete": 0,
        "payload_mismatch": 0,
        "rank_contract_missing": 0,
        "parse_errors": 0,
    }
    if trace_ok:
        replay_evidence = _read_replay_batched_refresh_evidence(
            args.route_trace,
            offset=route_trace_offset,
            expected_ranks=expected_replay_ranks,
        )
    generation_count = int(replay_evidence["publishes"])
    payloads_total = int(replay_evidence["payloads"])
    sentence_intents = int(replay_evidence["sentence_intents"])

    replay_defect_count = sum(
        int(replay_evidence[field])
        for field in (
            "malformed",
            "duplicates",
            "tp_incomplete",
            "payload_mismatch",
            "rank_contract_missing",
            "parse_errors",
        )
    )
    print(
        f"R1 世代活性: generation_count={generation_count} "
        f"payloads_total={payloads_total} "
        "source=replay_batched_route replay_enqueue_events="
        f"{replay_evidence['enqueue_events']} replay_world_generations="
        f"{replay_evidence['publishes']} replay_expected_ranks="
        f"{expected_replay_ranks}"
    )
    if replay_evidence["enqueue_events"] > 0 and replay_defect_count > 0:
        fails.append(
            "R1: replay route 证据不完整: "
            f"malformed={replay_evidence['malformed']} "
            f"duplicates={replay_evidence['duplicates']} "
            f"TP-incomplete={replay_evidence['tp_incomplete']} "
            f"payload-mismatch={replay_evidence['payload_mismatch']} "
            f"rank-contract-missing={replay_evidence['rank_contract_missing']} "
            f"json-parse-errors={replay_evidence['parse_errors']}"
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

    # ---- R3a 路由活性(冻结 RPC 结构化计数,主判据) ----
    compact_row_steps = -1
    route_rank_slots = (
        int(snapshot_counters["rank_slots"])
        if snapshot_counters is not None
        else 0
    )
    counter_kind4_missing = False
    if snapshot_ok:
        counters = snapshot_counters
        if counters is None:
            fails.append(
                "R3a: frozen route counter snapshot 不是非空的 "
                "80B-per-rank 完整记录"
                "(损坏/旧 64B 合同/未初始化)"
            )
        else:
            route_rank_slots = int(counters["rank_slots"])
            total = counters["total"]
            kind4 = counters["kind4"]
            compact_row_steps = int(counters.get("compact_row_steps", -1))
            ratio = (kind4 / total) if total else 0.0
            print(
                f"R3a 路由活性(frozen RPC): total={total} kind4={kind4} "
                f"({ratio:.1%}) has_rrp={counters['has_resolved_row_ptr']} "
                f"by_kind={counters['by_kind']} rank_slots={counters['rank_slots']}"
            )
            if not counters["rank_consistent"]:
                fails.append(
                    "R3a: TP rank-local route/liveness counters diverged; "
                    f"per_rank={counters['per_rank']}"
                )
            if (
                args.expected_route_counter_ranks > 0
                and route_rank_slots != args.expected_route_counter_ranks
            ):
                fails.append(
                    "R3a: frozen route snapshot 的 rank 数不匹配: "
                    f"observed={route_rank_slots} "
                    f"expected={args.expected_route_counter_ranks}"
                )
            counter_kind4_missing = bool(total > 0 and kind4 == 0)
            # ---- R3c 读侧活性(step build 侧计数,replay-aware 主判据) ----
            # FULL replay 对 Python route counter 不可见；proof specialization
            # 的 step counter 是 graph-independent 读侧真值。
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

    if args.baseline_compact_row_steps >= 0 and not snapshot_ok:
        fails.append(
            "R3c: 已请求本轮增量判定，但 frozen route counter snapshot 不可用"
        )
    elif args.baseline_compact_row_steps >= 0 and compact_row_steps < 0:
        fails.append(
            "R3c: frozen route counter snapshot 不含 replay-aware compact 读计数"
        )

    # ---- R3b fresh FULL replay authority: TP-complete replay hook only ----
    replay_route_authority = False
    if trace_ok:
        events = 0
        route_parse_errors: list[str] = []
        replay_hook_groups: dict[tuple[int, str], set[int]] = {}
        replay_hook_bad_groups: set[tuple[int, str]] = set()
        replay_hook_mismatches = 0
        replay_hook_duplicates = 0
        for evt in _iter_json_records(
            args.route_trace,
            offset=route_trace_offset,
            parse_errors=route_parse_errors,
        ):
            events += 1
            if evt.get("event") != (
                "mixed_page_full_cudagraph_replay_hook_check"
            ):
                continue
            captured_family = evt.get("captured_route_family")
            current_family = evt.get("current_route_family")
            family_mismatch = evt.get("route_family_mismatch")
            if (
                captured_family is None
                and current_family is None
                and family_mismatch is None
            ):
                # Bootstrap/runtime-mode probes have no bound graph yet.
                continue
            pid = evt.get("pid")
            step_id = evt.get("step_id")
            batch_descriptor = evt.get("batch_descriptor")
            if (
                type(pid) is not int
                or type(step_id) is not int
                or not isinstance(batch_descriptor, str)
                or not batch_descriptor
            ):
                replay_hook_mismatches += 1
                continue
            group_key = (step_id, batch_descriptor)
            if (
                captured_family != "resolved_row_ptr"
                or current_family != "resolved_row_ptr"
                or family_mismatch is not False
            ):
                replay_hook_mismatches += 1
                replay_hook_bad_groups.add(group_key)
                continue
            seen_pids = replay_hook_groups.setdefault(group_key, set())
            if pid in seen_pids:
                replay_hook_duplicates += 1
                replay_hook_bad_groups.add(group_key)
                continue
            seen_pids.add(pid)

        replay_hook_complete = 0
        replay_hook_incomplete = 0
        if replay_hook_groups and expected_replay_ranks > 0:
            for group_key, seen_pids in replay_hook_groups.items():
                if group_key in replay_hook_bad_groups:
                    continue
                if len(seen_pids) == expected_replay_ranks:
                    replay_hook_complete += 1
                else:
                    replay_hook_incomplete += 1
        elif replay_hook_groups:
            replay_hook_incomplete = len(replay_hook_groups)

        replay_route_authority = bool(
            replay_hook_complete > 0 and compact_row_steps > 0
            and replay_hook_incomplete == 0
            and replay_hook_mismatches == 0
            and replay_hook_duplicates == 0
            and not route_parse_errors
        )
        print(
            f"R3b FULL replay hook(trace): events={events} "
            f"replay_complete={replay_hook_complete} "
            f"replay_incomplete={replay_hook_incomplete} "
            f"replay_mismatch={replay_hook_mismatches} "
            f"replay_duplicates={replay_hook_duplicates}"
        )
        if route_parse_errors:
            fails.append(
                "R3b: fresh route window 含损坏 JSON: "
                f"count={len(route_parse_errors)} "
                f"sample={route_parse_errors[:4]}"
            )
        if (
            replay_hook_mismatches
            or replay_hook_duplicates
            or replay_hook_incomplete
        ):
            fails.append(
                "R3b: FULL replay route 证据不完整: "
                f"mismatch={replay_hook_mismatches} "
                f"duplicates={replay_hook_duplicates} "
                f"TP-incomplete={replay_hook_incomplete}"
            )
        if generation_count > 0 and not replay_route_authority:
            fails.append(
                "R3b: replay generation 缺少唯一的 TP-complete "
                "resolved_row_ptr hook + compact-step read authority"
            )
        if replay_route_authority:
            print(
                "[INFO] R3b: FULL replay route authority=TP-complete "
                "resolved_row_ptr hook + replay-aware compact step counter"
            )
    if counter_kind4_missing:
        if replay_route_authority:
            print(
                "[INFO] R3a: kind4 对 FULL replay 不可见；"
                "读侧以 R3b hook + R3c step counter 联合定谳"
            )
        else:
            fails.append(
                "R3a: frozen RPC kind4=0 且无 FULL replay read authority"
            )
    if not (snapshot_ok or trace_ok):
        warns.append(
            "R3 未判(--route-trace / --route-counter-snapshot 均不可用)"
        )

    # ---- R4 请求级 fallback：只把 mature decode 的不期望 dense 判死。 ----
    # prefill / policy-unready / refresh 与 threshold 前 mature dense 都是
    # 显式合法相位。唯一 fallback 定义是同一 request incarnation 已进入
    # compact 后又回到 dense/native 或 policy-unready。
    if step_trace_ok:
        compact_seen: set[tuple[int, str]] = set()
        request_lifecycle_seen: set[tuple[int, str]] = set()
        step_events = 0
        traced_rows = 0
        compact_rows = 0
        prefill_dense_rows = 0
        policy_unready_dense_rows = 0
        refresh_rows = 0
        precompact_dense_rows = 0
        post_compact_fallback_rows: list[str] = []
        lifecycle_missing_count = 0
        lifecycle_missing_samples: list[str] = []
        step_trace_parse_errors: list[str] = []
        tp_step_pids: set[int] = set()
        tp_step_groups: dict[int, tuple[str, int, set[int]]] = {}
        tp_parity_mismatches: list[str] = []
        tp_duplicate_events: list[str] = []
        tp_missing_fields: list[str] = []
        tp_parity_mismatch_count = 0
        tp_duplicate_event_count = 0
        tp_missing_field_count = 0
        mixed_phase_step_tokens: set[int] = set()
        mixed_compact_step_tokens: set[int] = set()
        mixed_contract_malformed_count = 0
        mixed_contract_malformed_samples: list[str] = []
        vector_fields = (
            "req_ids",
            "is_prefill_by_row",
            "row_policy_ready_by_row",
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
            "row_policy_ready_by_row",
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
            if args.require_mixed_prefill_decode:
                prefill_flags = vectors["is_prefill_by_row"]
                compact_flags = vectors["use_compact_by_row"]
                raw_prefill_count = event.get("prefill_row_count")
                raw_decode_count = event.get("decode_row_count")
                identity_token = event.get("step_identity_token")
                malformed_reason = ""
                if (
                    any(type(value) is not bool for value in prefill_flags)
                    or any(type(value) is not bool for value in compact_flags)
                ):
                    malformed_reason = "phase/compact vector contains non-bool"
                elif (
                    type(raw_prefill_count) is not int
                    or type(raw_decode_count) is not int
                ):
                    malformed_reason = "prefill/decode row counts are missing or non-int"
                elif type(identity_token) is not int:
                    malformed_reason = "step_identity_token is missing or non-int"
                else:
                    derived_prefill_count = sum(prefill_flags)
                    derived_decode_count = rows - derived_prefill_count
                    if (
                        raw_prefill_count != derived_prefill_count
                        or raw_decode_count != derived_decode_count
                    ):
                        malformed_reason = (
                            "prefill/decode row counts disagree with row vectors"
                        )
                    elif derived_prefill_count > 0 and derived_decode_count > 0:
                        mixed_phase_step_tokens.add(identity_token)
                        if any(compact_flags) and not all(compact_flags):
                            mixed_compact_step_tokens.add(identity_token)
                if malformed_reason:
                    mixed_contract_malformed_count += 1
                    if len(mixed_contract_malformed_samples) < 8:
                        mixed_contract_malformed_samples.append(
                            f"pid={pid} reason={malformed_reason}"
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
                    vectors["row_policy_ready_by_row"][row],
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
                is_prefill, row_policy_ready, use_compact, layer_refresh = (
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
                    request_lifecycle_seen.discard(key)
                    if row_mode not in (0, 2) or layer_refresh:
                        fails.append(
                            f"R4: req={req_id!r} prefill 相位非法 "
                            f"row_mode={row_mode} refresh={layer_refresh}"
                        )
                        continue
                    prefill_dense_rows += 1
                    request_lifecycle_seen.add(key)
                    continue
                if key not in request_lifecycle_seen:
                    lifecycle_missing_count += 1
                    if len(lifecycle_missing_samples) < 8:
                        lifecycle_missing_samples.append(
                            f"pid={pid},req={req_id}"
                        )
                    continue
                # Keep the judge's phase precedence identical to
                # resolve_decode_row_policy(): an explicit decode refresh is
                # legal before row-policy readiness because that refresh is
                # what materializes the first compact generation.
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
                if not row_policy_ready:
                    if row_mode != 0:
                        fails.append(
                            f"R4: req={req_id!r} policy-unready 相位非法 "
                            f"row_mode={row_mode} refresh={layer_refresh}"
                        )
                        continue
                    if args.reject_request_fallback and key in compact_seen:
                        post_compact_fallback_rows.append(
                            f"pid={pid},req={req_id},reason=policy_unready"
                        )
                    else:
                        policy_unready_dense_rows += 1
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

                if args.reject_request_fallback and key in compact_seen:
                    post_compact_fallback_rows.append(
                        f"pid={pid},req={req_id}"
                    )
                else:
                    precompact_dense_rows += 1

        if step_trace_parse_errors:
            sample = "; ".join(step_trace_parse_errors[:4])
            fails.append(
                "R4: request-level step trace 含损坏记录: "
                f"count={len(step_trace_parse_errors)} sample=[{sample}]"
            )
        if lifecycle_missing_count:
            fails.append(
                "R4: fresh step window 从请求中途开始，缺少 prefill 生命周期边界: "
                f"count={lifecycle_missing_count} "
                f"sample={lifecycle_missing_samples}"
            )

        print(
            f"R4 请求级相位({args.phase}): events={step_events} rows={traced_rows} "
            f"compact={compact_rows} prefill_dense={prefill_dense_rows} "
            f"policy_unready_dense={policy_unready_dense_rows} "
            f"refresh={refresh_rows} "
            f"precompact_or_short_dense={precompact_dense_rows} "
            f"fallback={len(post_compact_fallback_rows)}"
        )
        if require_step_trace and step_events == 0:
            fails.append("R4: 没有 fa3_step_state 请求级记录")
        if post_compact_fallback_rows:
            sample = ", ".join(post_compact_fallback_rows[:8])
            fails.append(
                "R4: 检出 post-compact mature decode dense/native fallback: "
                f"count={len(post_compact_fallback_rows)} sample=[{sample}]"
            )

        # ---- R5 TP step semantics: aggregate counters can prove a final
        # mismatch but cannot identify a transient compact/native split.  The
        # step identity is scheduler-derived and must carry byte-identical row
        # policy on every rank.  Runner offsets are captured before a request
        # and checked after its response, so every fresh group must be complete.
        expected_tp_ranks = (
            route_rank_slots if route_rank_slots > 0 else len(tp_step_pids)
        )
        if expected_tp_ranks > 1:
            incomplete_tokens: list[str] = []
            incomplete_token_count = 0
            if tp_step_groups:
                for token, (_digest, _reference_pid, seen_pids) in (
                    tp_step_groups.items()
                ):
                    if len(seen_pids) != expected_tp_ranks:
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
                f"incomplete={incomplete_token_count}"
            )
            if len(tp_step_pids) != expected_tp_ranks:
                fails.append(
                    "R5: step trace 的 TP PID 数与 frozen route snapshot 不一致: "
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
                    "R5: TP step trace 缺 rank 事件: "
                    f"count={incomplete_token_count} sample={incomplete_tokens}"
                )
            if not tp_step_groups:
                fails.append("R5: TP>1 但没有可比较的 step identity")
        elif step_events > 0:
            print(
                "R5 TP step 语义一致性: single-rank run, "
                f"observed_pids={sorted(tp_step_pids)}"
            )
        if args.require_mixed_prefill_decode:
            if expected_tp_ranks > 1:
                complete_mixed_phase_tokens = {
                    token
                    for token in mixed_phase_step_tokens
                    if token in tp_step_groups
                    and len(tp_step_groups[token][2]) == expected_tp_ranks
                }
                complete_mixed_compact_tokens = {
                    token
                    for token in mixed_compact_step_tokens
                    if token in tp_step_groups
                    and len(tp_step_groups[token][2]) == expected_tp_ranks
                }
            else:
                complete_mixed_phase_tokens = set(mixed_phase_step_tokens)
                complete_mixed_compact_tokens = set(mixed_compact_step_tokens)
            print(
                "R6 mixed prefill/decode: "
                f"phase_groups={len(complete_mixed_phase_tokens)} "
                f"compact_native_groups={len(complete_mixed_compact_tokens)} "
                f"malformed={mixed_contract_malformed_count}"
            )
            if mixed_contract_malformed_count:
                fails.append(
                    "R6: mixed step 合同字段损坏: "
                    f"count={mixed_contract_malformed_count} "
                    f"sample={mixed_contract_malformed_samples}"
                )
            if not complete_mixed_phase_tokens:
                fails.append(
                    "R6: 没有 TP 完整一致的 prefill+decode mixed step"
                )
            if not complete_mixed_compact_tokens:
                fails.append(
                    "R6: 没有 TP 完整一致且同时包含 compact/native row policy "
                    "的 prefill+decode mixed step"
                )
    elif require_step_trace:
        fails.append("R4: 请求了 request-level 判定但 step trace 不可用")

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
