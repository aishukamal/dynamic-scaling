#!/usr/bin/env python3
"""Controller-run analysis: per-step table joining the orchestrator log (step wall/wait/train/sync, FTSNAP states) with
the controller JSONL (decisions, actuations), chip-seconds per step with chips HELD (4 trainer + 4 per rollout JobSet
whose pod existed during the step, i.e. live states incl. INITIALIZING/PENDING), phase means, and the static-ladder
comparison (TUNIX-BASELINE r1/r2/r3: 427/463/621 chip-s per step).
Usage: ctl_analyze.py <orch.log (kubectl --timestamps)> <controller.jsonl> [--csv out.csv]"""
import re, sys, json, collections, datetime, statistics as st
log, ctl = sys.argv[1], sys.argv[2]
csv_out = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
re_metric = re.compile(r"METRIC step=(\d+) (\S+)=(\S+)")
metrics = collections.defaultdict(dict); ft_end = {}; step_end_t = {}; step_begin_t = {}
events = []
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
    m = re.search(r"<<< Step (\d+) finished", line)
    if m: step_end_t[int(m.group(1))] = t; continue
    m = re.search(r">>> Step (\d+) starting", line)
    if m: step_begin_t[int(m.group(1))] = t; continue
    m = re.search(r"FTSNAP step=(\d+) t=\S+ active_rollout_workers=(\d+) states=(\{[^}]*\}) ft=(\{[^}]*\})", line)
    if m:
        ft_end[int(m.group(1))] = (int(m.group(2)), eval(m.group(3)), eval(m.group(4))); continue
    m = re.search(r"\[rollout-ft\] action=(join|rejoin|promote|evict|leave) worker(?:_id)?=(\S+)(.*)", line)
    if m and t: events.append((t, m.group(1), m.group(2), m.group(3).strip()[:80]))
# per-step live rollout JobSets = workers whose state at step end is not evicted, plus workers that were live at step start (pod existed)
decisions = []; actuations = []; dones = []; guard = []; ext = []
for l in open(ctl):
    try: r = json.loads(l)
    except Exception: continue
    d = r.get("decision")
    if d and d["action"] != "hold": decisions.append((r["t"], d))
    if "actuation" in r: actuations.append((r["t"], r["actuation"]))
    if "actuation_done" in r: dones.append((r["t"], r["actuation_done"]))
    if "guard_eval" in r: guard.append((r["t"], r["guard_eval"]))
    if "membership_changed_externally" in r: ext.append((r["t"], r["membership_changed_externally"]))
def f(x, p=1): return "-" if x is None else f"{x:.{p}f}"
def find(d, suf):
    for k, v in d.items():
        if k.endswith(suf): return v
rows = []
# live-hosts timeline from events: a worker holds its host from (re)join (pod exists ~30 s earlier) until evict (pod gone shortly after)
# approximate held hosts during step s as the max of live rollout workers at step start FTSNAP and step end FTSNAP (states != evicted)
prev_states = None
print("step | end_at | wall | wait | ratio | train | sync | live_rollout_at_end | held_chips | chip_s | retries | evictions")
for s in sorted(metrics):
    d = metrics[s]; wall = find(d, "orchestrator/train/step_time_sec"); wait = find(d, "orchestrator/train/exposed_generation_time")
    train = find(d, "orchestrator/train/policy_training_time"); sync = find(d, "orchestrator/train/weight_sync_time")
    toks = find(d, "rollout/train/global_valid_toks")
    snap = ft_end.get(s); live = None; ret = ev = None
    if snap:
        states = snap[1]; live = sorted(w for w, v in states.items() if w != "te-train" and not w.endswith("-train") and v != "evicted")
        ret = snap[2].get("rollout_retries_total"); ev = snap[2].get("rollout_worker_evictions_total")
    nlive = len(live) if live is not None else None
    held = (4 + 4 * nlive) if nlive is not None else None
    chip_s = held * wall if (held and wall) else None
    rows.append(dict(step=s, end=step_end_t.get(s), wall=wall, wait=wait, ratio=(wait / wall if wall else None), train=train, sync=sync, live=nlive, held=held, chip_s=chip_s, toks=toks, retries=ret, evictions=ev))
    end = datetime.datetime.fromtimestamp(step_end_t[s], datetime.timezone.utc).strftime("%H:%M:%S") if s in step_end_t else "-"
    print(f"{s} | {end} | {f(wall)} | {f(wait)} | {f(wait/wall if wall else None,3)} | {f(train,2)} | {f(sync,2)} | {live} | {held} | {f(chip_s,0)} | {ret} | {ev}")
# phases
def phase(name, sel):
    rs = [r for r in rows if sel(r) and r["wall"]]
    if not rs: return
    print(f"{name}: n={len(rs)} wall mean {st.mean(r['wall'] for r in rs):.1f} (range {min(r['wall'] for r in rs):.1f}-{max(r['wall'] for r in rs):.1f}) wait mean {st.mean(r['wait'] for r in rs):.1f} ratio mean {st.mean(r['ratio'] for r in rs):.3f} chip-s/step mean {st.mean(r['chip_s'] for r in rs if r['chip_s']):.0f}")
print()
phase("steps with 1 live rollout host (excl. step 0)", lambda r: r["live"] == 1 and r["step"] > 0)
phase("steps with 2 live rollout hosts", lambda r: r["live"] == 2)
phase("steps with 3 live rollout hosts", lambda r: r["live"] == 3)
tot = [r for r in rows if r["chip_s"] and r["step"] > 0]
if tot:
    cs = sum(r["chip_s"] for r in tot); n = len(tot); wall = sum(r["wall"] for r in tot)
    print(f"ALL steps 1..{max(r['step'] for r in tot)}: n={n} wall sum {wall:.0f} s, chip-s sum {cs:.0f}, chip-s/step {cs/n:.0f}, steps per chip-hour {3600*n/cs:.2f}; static arms (baseline): r1 427 chip-s/step = {3600/427:.2f} steps/chip-h, r2 463 = {3600/463:.2f}, r3 621 = {3600/621:.2f}")
print("\nDECISIONS:");  [print(datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S"), d["action"], "step", d["inputs"]["param_version"], "ratio %.3f" % d["inputs"]["trainer_idle_ratio"], "active", d["inputs"]["active"], "->", d["desired"], ":", d["reason"]) for t, d in decisions]
print("\nACTUATIONS:"); [print(datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S"), json.dumps(a)[:220]) for t, a in actuations]
print("\nCOMPLETED:"); [print(datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S"), {k: a.get(k) for k in ("kind", "worker_id", "result", "join_latency_s", "evict_latency_s", "requeued", "step_at_start", "step_at_end", "active_after")}) for t, a in dones]
print("\nGUARD:"); [print(datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S"), {k: g.get(k) for k in ("worker_id", "before_steps", "before_wait_s", "after_steps", "after_wait_s", "gain", "blocked")}) for t, g in guard]
print("\nEXTERNAL MEMBERSHIP CHANGES:"); [print(datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S"), e) for t, e in ext]
print("\nMEMBERSHIP EVENTS (orchestrator):"); [print(datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3], a, w, x) for t, a, w, x in events]
if csv_out:
    import csv
    with open(csv_out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); [wr.writerow(r) for r in rows]
