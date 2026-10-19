#!/usr/bin/env bash
# dyn3_status.sh [job] : compact status from the local partial pool log (copied by stop_after every 30 s) + controller log
JOB=${1:-tpum3-dyn3}; D="$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3"
python3 - "$D/results/pool-$JOB.partial.jsonl" "$D/logs/controller-$JOB.out" <<'PY'
import json, sys, datetime
pool, ctl = sys.argv[1], sys.argv[2]
recs = [json.loads(l) for l in open(pool) if l.startswith("{")]
import os
cut = os.environ.get("CUT", "2026-10-16T19:20")
recs = [r for r in recs if r.get("t", "") >= cut]
for r in recs:
    k = r["kind"]
    if k in ("lb", "health_probe_failed", "sync_begin"): continue
    t = r["t"][11:19]
    if k == "step":
        m = r["metrics"]; u = r.get("util") or {}
        print(f"{t} STEP v{r['step']} step_s={m.get('timing_s/step',0):.0f} wait={m.get('timing_s/gen',0):.0f} sync={m.get('timing_s/update_weights',0):.1f} active={r['active']} slot_util={u.get('decode_slot_util',0):.3f} reward={m.get('critic/score/mean')}")
    elif k == "sync_end":
        print(f"{t} sync_end v{r.get('global_steps')} sync_s={r['sync_s']:.1f} replicas={r.get('manager_replicas')}")
    elif k in ("spawn_start", "spawn_retry", "spawn_ready", "join_on_ready", "join_on_ready_error", "join_on_ready_skipped", "early_spawn", "replacement_spawn", "dead_detected", "dead_removed", "destroy", "trainer_blocked_by_join", "sync_waited_for_join", "hold", "spawn_error", "spawn_refused", "leave", "join_before_sync", "join_after_sync", "repair_error", "desired_error"):
        keep = {a: (round(b, 2) if isinstance(b, float) else b) for a, b in r.items() if a in ("rank", "version", "attempts", "init_standalone_s", "wait_clean_weights_s", "joiner_sync_s", "lb_add_s", "req_to_serving_s", "ready_to_serving_s", "pod_ray_s", "engine_init_s", "actor_busy_at_request", "waited_s", "desired", "have", "active", "error", "sync_error", "why", "total_s", "reason", "address", "phases", "req_to_start_s")}
        print(f"{t} {k} {keep}")
n_probe = sum(1 for r in recs if r["kind"] == "health_probe_failed")
print(f"health_probe_failed count: {n_probe}")
try:
    for l in open(ctl):
        if not l.startswith("{"): continue
        d = json.loads(l)
        if d["kind"] in ("out", "actuator", "actuator_wait", "in", "decision"):
            t = datetime.datetime.utcfromtimestamp(d["t"]).strftime("%H:%M:%S")
            print(f"{t} CTL {d['kind']} {json.dumps({a: b for a, b in d.items() if a not in ('t', 'kind')})[:300]}")
    last = [json.loads(l) for l in open(ctl) if l.startswith("{")][-1]
    t = datetime.datetime.utcfromtimestamp(last["t"]).strftime("%H:%M:%S")
    print(f"{t} CTL last {last['kind']} {json.dumps({a: b for a, b in last.items() if a not in ('t', 'kind')})[:400]}")
except FileNotFoundError:
    print("no controller log")
PY
date -u +%FT%TZ
