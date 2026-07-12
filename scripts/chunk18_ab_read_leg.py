#!/usr/bin/env python3
"""CHUNK18 AB 腿判读:tps/8-hash/counts/capture/arena 一体化对账。用法: read_leg.py <summary.json> [tkg.log]"""
import hashlib, json, sys, re
from collections import Counter, defaultdict

# Final merged baseline anchors (2026-07-12). Chunk size changes refresh
# completion timing, so the 4B speed tier intentionally has one stable anchor
# per arm; the 0.6B semantic golden remains shared.
ANCHOR_8_BY_CHUNK = {
    14: {"fff0445e", "c0a6376d", "99a9aea2", "7db6f96d", "fda56972", "deef29a8", "8e8e2dca", "8778b158"},
    18: {"970c061e", "c0a6376d", "99a9aea2", "03eb3325", "fda56972", "deef29a8", "f6d008c1", "4076e3d5"},
}
GOLDEN = {"ad585b37", "44e80946"}

s = json.load(open(sys.argv[1]))
prov = s.get("run_provenance") or {}
capture_chunk = int(
    prov.get("capture_chunk_effective", prov.get("capture_chunk", 0)) or 0
)
print(f"git_head={str(prov.get('git_head'))[:9]} capture_chunk_effective={capture_chunk} model={str(prov.get('model')).rsplit('/',1)[-1]} fa3_so={str(s.get('fa3_so_sha256'))[:12]}")
print(f"all_decode_tps={s.get('all_decode_tps')} decode_tps={s.get('decode_tps')} p50_us={round(s.get('decode_p50_us') or -1,1)} p95_us={round(s.get('decode_p95_us') or -1,1)}")
print(f"gate={s.get('gate_passed')} prod_gate={s.get('production_gate_passed')} fallback={s.get('dense_native_fallback_count')} route_proof={s.get('route_proof_passed')} speed_child_route_proof={s.get('speed_child_route_proof_passed')}")
print(f"counts={s.get('refresh_reason_counts')} payloads={s.get('refresh_payloads')} intents={s.get('refresh_trigger_intents')}")
print(f"arena_reserved={s.get('arena_reserved_bytes')} peak={s.get('arena_peak_bytes')} budget_exceeded={s.get('arena_budget_exceeded')} bind={s.get('arena_bind_status')}")

# 8-hash (outputs dict: req_id -> {text, token_ids})
op = s.get("outputs_path")
if op:
    try:
        outs = json.load(open(op))
        items = sorted(outs.items()) if isinstance(outs, dict) else list(enumerate(outs))
        hs = [hashlib.sha256(str(v.get("token_ids")).encode()).hexdigest()[:8] for _, v in items]
        anchor = GOLDEN if len(hs) == 2 else ANCHOR_8_BY_CHUNK.get(capture_chunk, set())
        hits = sum(1 for h in hs if h in anchor)
        texts_ok = all(isinstance(v.get("text"), str) and len(v.get("text")) > 20 for _, v in items)
        print(f"hashes={hs}")
        anchor_label = "GOLDEN" if anchor is GOLDEN else f"8HASH_CHUNK{capture_chunk}"
        print(f"anchor_match={hits}/{len(hs)} ({anchor_label}) text_nonempty={texts_ok}")
    except Exception as e:
        print(f"outputs read fail: {e}")

# capture count from refresh profile (last flush per pid)
rp = s.get("refresh_profile_path")
if rp:
    try:
        cap, rep = {}, {}
        for line in open(rp):
            if "refresh.flush" not in line:
                continue
            pid, _, payload = line.split("\t", 2)
            d = json.loads(payload)
            cap[pid] = d.get("selector_topk_graph_capture_count")
            rep[pid] = d.get("selector_topk_graph_replay_count")
        print(f"flush_capture_count={cap} replay={rep}")
    except Exception as e:
        print(f"refresh_profile read fail: {e}")

# tkg census: capture keys per pid + republish layer-group tags
if len(sys.argv) > 2:
    try:
        cap_keys = defaultdict(list)
        repub = Counter()
        for line in open(sys.argv[2]):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and "capture#" in parts[1]:
                cap_keys[parts[0]].append(parts[2])
            elif len(parts) >= 3 and parts[1] == "republish":
                m = re.search(r"layers_(\d+)_(\d+)_n(\d+)", parts[2])
                if m:
                    repub[(parts[0], m.group(0))] += 1
        for pid, keys in sorted(cap_keys.items()):
            shapes = Counter()
            for k in keys:
                t = eval(k)
                if len(t) == 18:
                    shapes[f"topk_chunk_n{t[0]}"] += 1
                elif len(t) == 22:
                    shapes[f"aux_layers_{t[-3]}_{t[-2]}_n{t[5]}"] += 1
                else:
                    shapes[f"len{len(t)}"] += 1
            print(f"tkg pid={pid} captures={len(keys)} shapes={dict(shapes)}")
        by_pid = defaultdict(dict)
        for (pid, tag), n in sorted(repub.items()):
            by_pid[pid][tag] = n
        for pid, tags in sorted(by_pid.items()):
            print(f"republish pid={pid} groups={tags}")
    except FileNotFoundError:
        print("tkg log missing")
