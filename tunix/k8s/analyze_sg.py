#!/usr/bin/env python3
"""Slot-gate run analysis: per-step table joining the orchestrator log (step wall / wait / FTSNAP live workers /
SANDBOXSNAP pods) with the controller JSONL (per-step slot utilization, band-only verdict, decisions, actuations).
Usage: analyze_sg.py <orch.log (kubectl --timestamps)> <controller.jsonl> [--csv out.csv] [--prefix tg1]"""
import re, sys, json, collections, datetime, statistics as st
log, ctl = sys.argv[1], sys.argv[2]
csv_out = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
pfx = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "tg"
re_metric = re.compile(r"METRIC step=(\d+) (\S+)=(\S+)")
metrics = collections.defaultdict(dict); ft_end = {}; step_end_t = {}; sbx = collections.defaultdict(list); events = []
def ts(line):
    m = re.match(r"^(\S+?)(\.\d+)?Z\s", line)
    if not m: return None
    base = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
    return base + (float("0" + m.group(2)) if m.group(2) else 0.0)
for line in open(log, errors="replace"):
    t = ts(line)
    m = re_metric.search(line)
    if m:
        try: metrics[int(m.group(1))][m.group(2)] = float(m.group(3))
        except ValueError: pass
        continue
    m = re.search(r"<<< (?:DeepSWE )?[Ss]tep (\d+) finished", line)
    if m: step_end_t[int(m.group(1))] = t; continue
    m = re.search(r"FTSNAP step=(\d+) t=\S+ active_rollout_workers=(\d+) states=(\{[^}]*\}) ft=(\{[^}]*\})", line)
    if m: ft_end[int(m.group(1))] = (int(m.group(2)), eval(m.group(3)), eval(m.group(4))); continue
    m = re.search(r"SANDBOXSNAP step=(\d+) t=(\S+) pods_total=(\d+) pods=(\{[^}]*\}) pods_ready=(\d+) warmpools=(\d+) pool_replicas=(\d+) pool_available=(\d+) claims=(\d+)", line)
    if m:
        d = eval(m.group(4)); sbx[int(m.group(1))].append(dict(t=float(m.group(2)), total=int(m.group(3)), pending=d.get("Pending", 0), running=d.get("Running", 0), ready=int(m.group(5)), pools=int(m.group(6)), claims=int(m.group(9)))); continue
    m = re.search(r"\[rollout-ft\] action=(join|rejoin|promote|evict|leave) worker(?:_id)?=(\S+)(.*)", line)
    if m and t: events.append((t, m.group(1), m.group(2), m.group(3).strip()[:80]))
# controller JSONL
step_util = {}; gate_at_step = {}; decisions = []; actuations = []; dones = []; guard = []; ext = []; start = None
for l in open(ctl):
    try: r = json.loads(l)
    except Exception: continue
    if r.get("kind") == "start": start = r; continue
    su = r.get("step_util")
    if su:
        for x in (su if isinstance(su, list) else [su]): step_util[x["step"]] = x
    d = r.get("decision"); g = r.get("gate")
    if d and d["counters"].get("new_sample") and g is not None: gate_at_step[d["inputs"]["param_version"]] = (g, d)
    if d and d["action"] != "hold": decisions.append((r["t"], d, g))
    if "actuation" in r: actuations.append((r["t"], r["actuation"]))
    if "actuation_done" in r: dones.append((r["t"], r["actuation_done"]))
    if "guard_eval" in r: guard.append((r["t"], r["guard_eval"]))
    if "membership_changed_externally" in r: ext.append((r["t"], r["membership_changed_externally"]))
def f(x, p=1): return "-" if x is None else f"{x:.{p}f}"
def find(d, suf):
    for k, v in d.items():
        if k.endswith(suf): return v
def hms(t): return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S") if t else "-"
rows = []
print("step | end_at | wall | wait | share | train | sync | util_peak | util_mean | infl_peak/cap | pend_q | live | held | chip_s | sbx_pending(begin/end) | sbx_total(end) | band | verdict | decision")
for s in sorted(metrics):
    d = metrics[s]; wall = find(d, "orchestrator/train/step_time_sec"); wait = find(d, "orchestrator/train/exposed_generation_time")
    train = find(d, "orchestrator/train/policy_training_time"); sync = find(d, "orchestrator/train/weight_sync_time")
    toks = find(d, "rollout/train/global_valid_toks"); reward = find(d, "rewards/train/mean")
    snap = ft_end.get(s); live = None; ret = ev = None
    if snap:
        states = snap[1]; live = sorted(w for w, v in states.items() if not w.endswith("-train") and v != "evicted")
        ret = snap[2].get("rollout_retries_total"); ev = snap[2].get("rollout_worker_evictions_total")
    nlive = len(live) if live is not None else None
    held = (4 + 4 * nlive) if nlive is not None else None
    chip_s = held * wall if (held and wall) else None
    u = step_util.get(s, {}); gd = gate_at_step.get(s); g = gd[0] if gd else {}; dec = gd[1] if gd else {}
    sb = sbx.get(s, []); sb_begin = sb[0]["pending"] if sb else None; sb_end = sb[-1]["pending"] if len(sb) > 1 else None; sb_tot = sb[-1]["total"] if sb else None
    # membership events (join/promote/evict/leave/rejoin, orchestrator clock) that fell inside this step: (prev step end, this step end]
    t_end_s = step_end_t.get(s); t_prev = step_end_t.get(s - 1)
    evs = [f"{hms(t)} {a} {w.split('/')[-1][:24]}" for t, a, w, x in events if t_end_s and (t_prev is None or t > t_prev) and t <= t_end_s and a in ("promote", "evict", "leave", "rejoin")]
    ctl_ev = [f"{hms(t)} ctl:{d['action']}->{d['desired']}" for t, d, g in decisions if t_end_s and (t_prev is None or t > t_prev) and t <= t_end_s]
    rows.append(dict(step=s, end=hms(step_end_t.get(s)), wall=wall, wait=wait, share=(wait / wall if wall else None), train=train, sync=sync,
                     util_peak=u.get("peak"), util_mean=u.get("mean"), inflight_peak=u.get("inflight_peak"), cap=u.get("cap"), pending_peak=u.get("pending_peak"), n_samples=u.get("n_samples"),
                     live=nlive, held=held, chip_s=chip_s, sbx_pending_begin=sb_begin, sbx_pending_end=sb_end, sbx_total_end=sb_tot, sbx_pending_max=(max(x["pending"] for x in sb) if sb else None),
                     band=g.get("band"), verdict=g.get("verdict"), slot_util_window=g.get("slot_util_window"), decision=dec.get("action"), reason=dec.get("reason"), toks=toks, reward=reward, retries=ret, evictions=ev,
                     events="; ".join(ctl_ev + evs),
                     sandbox_confounded=("yes" if any((x["pending"] or 0) > 0 for x in sb) else ("no" if sb else ""))))
    r = rows[-1]
    print(f"{s} | {r['end']} | {f(wall)} | {f(wait)} | {f(r['share'],3)} | {f(train,2)} | {f(sync,2)} | {f(r['util_peak'],3)} | {f(r['util_mean'],3)} | {r['inflight_peak']}/{r['cap']} | {r['pending_peak']} | {nlive} | {held} | {f(chip_s,0)} | {sb_begin}/{sb_end} | {sb_tot} | {r['band']} | {r['verdict']} | {r['decision']}:{(r['reason'] or '')[:50]}")
def phase(name, sel):
    rs = [r for r in rows if sel(r) and r["wall"]]
    if not rs: return
    print(f"{name}: n={len(rs)} wall mean {st.mean(r['wall'] for r in rs):.1f} (range {min(r['wall'] for r in rs):.1f}-{max(r['wall'] for r in rs):.1f}) wait mean {st.mean(r['wait'] for r in rs):.1f} share mean {st.mean(r['share'] for r in rs):.3f} util_peak mean {st.mean(r['util_peak'] for r in rs if r['util_peak'] is not None) if any(r['util_peak'] is not None for r in rs) else float('nan'):.3f} chip-s/step mean {st.mean(r['chip_s'] for r in rs if r['chip_s']):.0f} = {3600/st.mean(r['chip_s'] for r in rs if r['chip_s']):.2f} steps/chip-h")
print()
phase("steps 1-7 (matched to the static ladder / te)", lambda r: 1 <= r["step"] <= 7)
phase("steps with 1 live rollout host (excl. step 0)", lambda r: r["live"] == 1 and r["step"] > 0)
phase("steps with 2 live rollout hosts", lambda r: r["live"] == 2)
phase("steps with 3 live rollout hosts", lambda r: r["live"] == 3)
tot = [r for r in rows if r["chip_s"] and r["step"] > 0]
if tot:
    cs = sum(r["chip_s"] for r in tot); n = len(tot); wall = sum(r["wall"] for r in tot)
    print(f"ALL steps 1..{max(r['step'] for r in tot)}: n={n} wall sum {wall:.0f} s, chip-s sum {cs:.0f}, chip-s/step {cs/n:.0f}, steps per chip-hour {3600*n/cs:.2f}")
print("\nSTART:", json.dumps({k: start["policy"][k] for k in ("idle_hi", "idle_lo", "dwell_steps", "min_action_interval_s", "max_standalone", "util_hi", "util_lo")}) if start else "-", json.dumps({k: start["run"].get(k) for k in ("util_stat", "util_window", "poll_interval_s", "guard_enabled", "guard_window", "guard_min_gain")}) if start else "")
print("\nNON-HOLD DECISIONS:"); [print(hms(t), d["action"], "step", d["inputs"]["param_version"], "share %.3f" % d["inputs"]["trainer_idle_ratio"], "slot_util", f(d["inputs"].get("slot_util"), 3), "active", d["inputs"]["active"], "->", d["desired"], ":", d["reason"], "| gate:", (g or {}).get("verdict")) for t, d, g in decisions]
print("\nACTUATIONS:"); [print(hms(t), json.dumps(a)[:220]) for t, a in actuations]
print("\nCOMPLETED:"); [print(hms(t), {k: a.get(k) for k in ("kind", "worker_id", "result", "join_latency_s", "evict_latency_s", "requeued", "step_at_start", "step_at_end", "active_after")}) for t, a in dones]
print("\nGUARD:"); [print(hms(t), {k: g.get(k) for k in ("worker_id", "before_steps", "before_wait_s", "after_steps", "after_wait_s", "gain", "blocked")}) for t, g in guard]
print("\nEXTERNAL MEMBERSHIP CHANGES:"); [print(hms(t), e) for t, e in ext]
verd = collections.Counter(g.get("verdict") for g, d in gate_at_step.values())
print("\nGATE VERDICTS per completed step:", dict(verd))
print("\nMEMBERSHIP EVENTS (orchestrator):"); [print(datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3], a, w, x) for t, a, w, x in events]
if csv_out:
    import csv
    with open(csv_out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); [wr.writerow(r) for r in rows]
