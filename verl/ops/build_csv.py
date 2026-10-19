#!/usr/bin/env python3
"""build_csv.py <pool jsonl> <sandbox jsonl or -> <out csv> : per-version rows for the coordinator's timeline plot.

Columns: version, t_end (UTC of the step log), step_s, wait_s (timing_s/gen), idle_ratio (wait/step), train_s
(old_log_prob + ref + adv + update_actor as logged), sync_s (update_weights), active_replicas (at the step log),
n_active_mean (time-weighted over the window), slot_util (time-weighted decode-slot occupancy = LB in-flight / (active x
max_num_seqs), clamped to 1), share_slots_full, lb_inflight_mean, traj_inflight (at the step log), reward, resp_len,
sandbox_pending (max unschedulable Pending among OUR swe- sandbox pods in the window), sandbox_running (max), sandbox_waiting (max ContainerCreating),
held_chips (4 trainer + 4 x time-weighted active samplers), chip_s (held_chips x step_s),
events (join/leave/dead/spawn records in the window with UTC times).
"""
import csv, json, os, sys
TRAINER_CHIPS = int(os.environ.get("TRAINER_CHIPS", "4"))  # 4 for the 1.7B runs (FSDP=4 host), 8 for the RECIPE2 8-chip trainer
from datetime import datetime, timezone

def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

pool, sbx, out = sys.argv[1], sys.argv[2], sys.argv[3]
recs = []
for line in open(pool):
    line = line.strip()
    if line.startswith("{"):
        try: recs.append(json.loads(line))
        except json.JSONDecodeError: pass
steps = sorted([r for r in recs if r.get("kind") == "step"], key=lambda r: r["step"])
events = [r for r in recs if r.get("kind") in ("join_before_sync", "join_after_sync", "leave", "dead_detected", "dead_removed", "spawn_start", "spawn_ready", "spawn_joined", "join_on_ready", "join_on_ready_error", "join_on_ready_skipped", "replacement_spawn", "destroy", "trainer_blocked_by_join", "sync_waited_for_join", "early_spawn")]
sb = []
if sbx != "-":
    for line in open(sbx):
        try: sb.append(json.loads(line))
        except json.JSONDecodeError: pass
rows = []
prev_t = None
for r in steps:
    m = r["metrics"]; t_end = ts(r["t"])
    step_s = m.get("timing_s/step") or 0.0
    t_start = prev_t if prev_t is not None else t_end - step_s
    prev_t = t_end
    win = [e for e in events if t_start <= ts(e["t"]) <= t_end]
    sbw = [x for x in sb if t_start <= ts(x["t"]) <= t_end]
    # ours only (name prefix swe-): unschedulable Pending = Pending phase minus pods merely in ContainerCreating
    def _swe(x, key): return (x.get("swe") or {}).get(key, 0)
    def _swe_cc(x): return (x.get("swe_waiting") or {}).get("ContainerCreating", 0)
    # unscheduled Pending (no nodeName) when the monitor recorded it; older samples fall back to Pending minus ContainerCreating
    pend = max([(x["swe_unscheduled"] if "swe_unscheduled" in x else max(0, _swe(x, "Pending") - _swe_cc(x))) for x in sbw], default=None)
    run = max([_swe(x, "Running") for x in sbw], default=None)
    wait_cc = max([_swe_cc(x) for x in sbw], default=None)
    util = r.get("util") or {}; traj = r.get("traj") or {}
    gen = m.get("timing_s/gen") or 0.0
    train = sum(float(m.get(k) or 0.0) for k in ("timing_s/old_log_prob", "timing_s/ref", "timing_s/adv", "timing_s/update_actor"))
    rows.append({
        "version": r["step"], "t_end": r["t"][:19] + "Z", "step_s": round(step_s, 2), "wait_s": round(gen, 2),
        "idle_ratio": round(gen / step_s, 3) if step_s else None, "train_s": round(train, 2),
        "sync_s": round(float(m.get("timing_s/update_weights") or 0.0), 3), "active_replicas": r.get("n_active"),
        "n_active_mean": round(util["n_active_mean"], 2) if "n_active_mean" in util else None,
        "slot_util": round(util["decode_slot_util"], 3) if "decode_slot_util" in util else None,
        "share_slots_full": round(util["share_slots_full"], 3) if "share_slots_full" in util else None,
        "lb_inflight_mean": round(util["lb_inflight_mean"], 1) if "lb_inflight_mean" in util else None,
        "traj_inflight": traj.get("traj_inflight"), "reward": m.get("critic/score/mean"), "resp_len": m.get("response_length/mean"),
        "sandbox_pending": pend, "sandbox_confounded": (1 if (pend or 0) > 0 else 0) if pend is not None else None,
        "sandbox_running": run, "sandbox_waiting": wait_cc,
        "held_chips": round(TRAINER_CHIPS + 4 * util["n_active_mean"], 2) if "n_active_mean" in util else None,
        "chip_s": round((TRAINER_CHIPS + 4 * util["n_active_mean"]) * step_s, 1) if "n_active_mean" in util else None,
        "events": "; ".join(f"{e['t'][11:19]} {e['kind']}" + (f" r{e['rank']}" if e.get('rank') is not None else "") for e in win),
    })
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["version"])
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows -> {out}")
for r in rows:
    print(r["version"], r["step_s"], r["wait_s"], r["idle_ratio"], r["train_s"], r["sync_s"], r["active_replicas"], r["slot_util"], r["sandbox_pending"], r["events"])
