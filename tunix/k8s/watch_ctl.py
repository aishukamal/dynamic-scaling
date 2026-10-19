#!/usr/bin/env python3
"""Summarize the controller JSONL (stdin or file): per-step trajectory, non-hold decisions, actuations, guard evals,
external membership changes. Usage: watch_ctl.py <decisions.jsonl> [--all]"""
import json, sys, datetime
path = sys.argv[1]; show_all = "--all" in sys.argv
def ts(t): return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S")
steps = {}; events = []
for line in open(path):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("kind") == "start": events.append((r["t"], "START policy=%s" % {k: r["policy"][k] for k in ("idle_hi","idle_lo","dwell_steps","min_action_interval_s","max_standalone")})); continue
    st = r.get("status") or {}
    s = st.get("last_completed_step")
    if s is not None and s not in steps:
        steps[s] = (r["t"], st.get("step_time_sec"), st.get("exposed_generation_time"), st.get("trainer_wait_ratio"), st.get("counts", {}).get("active"), st.get("tokens_per_sec"), st.get("ft", {}).get("rollout_retries_total"), st.get("ft", {}).get("rollout_worker_evictions_total"), r.get("decision", {}).get("counters"))
    d = r.get("decision")
    if d and d["action"] != "hold": events.append((r["t"], f"DECISION {d['action']} -> desired {d['desired']} at step {d['inputs']['param_version']} ratio {d['inputs']['trainer_idle_ratio']:.3f} active {d['inputs']['active']}: {d['reason']}"))
    if "actuation" in r: events.append((r["t"], f"ACTUATION {json.dumps(r['actuation'])[:300]}"))
    if "actuation_done" in r:
        a = r["actuation_done"]; events.append((r["t"], f"DONE {a['kind']} {a['worker_id']} result={a['result']} " + (f"join_latency={a.get('join_latency_s'):.1f}s" if a.get('join_latency_s') is not None else f"evict_latency={a.get('evict_latency_s')} requeued={a.get('requeued')}") + f" step_start={a.get('step_at_start')} step_end={a.get('step_at_end')} active_after={a.get('active_after')}"))
    if "guard_eval" in r: events.append((r["t"], f"GUARD {json.dumps(r['guard_eval'])[:300]}"))
    if "membership_changed_externally" in r: events.append((r["t"], f"EXTERNAL {json.dumps(r['membership_changed_externally'])}"))
    if "policy_reset" in r: events.append((r["t"], "POLICY_RESET"))
    if "error" in r and show_all: events.append((r["t"], "ERROR " + r["error"]))
print("step | seen_at | wall | wait | ratio | active | tok/s | retries | evictions | counters")
for s in sorted(steps):
    t, wall, wait, ratio, act, tps, ret, ev, c = steps[s]
    f = lambda x, p=1: "-" if x is None else f"{x:.{p}f}"
    print(f"{s} | {ts(t)} | {f(wall)} | {f(wait)} | {f(ratio,3)} | {act} | {f(tps,0)} | {ret} | {ev} | hi={c.get('hi_count') if c else '-'} lo={c.get('lo_count') if c else '-'}")
print("\nEVENTS:")
for t, e in events: print(ts(t), e)
