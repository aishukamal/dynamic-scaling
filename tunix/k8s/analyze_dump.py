#!/usr/bin/env python3
"""Per-step tool-call validity from the rollout-side dump + orchestrator log.
Usage: analyze_dump.py <dump-dir-with-turns-*.jsonl> <orch log (kubectl --timestamps)> <rollout log>"""
import json, glob, sys, re, collections, datetime
dump, orch, roll = sys.argv[1], sys.argv[2], sys.argv[3]
def _load(paths):
    out=[]; bad=0
    for f in paths:
        for l in open(f, errors="replace"):
            if not l.strip(): continue
            try: out.append(json.loads(l))
            except Exception: bad+=1
    if bad: print(f"# skipped {bad} undecodable dump lines")
    return out
recs = _load(sorted(glob.glob(dump + "/*.jsonl")))
def ts(line):
    m = re.match(r"^(\S+?)(\.\d+)?Z\s", line)
    if not m: return None
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp() + (float("0" + m.group(2)) if m.group(2) else 0.0)
# batch dispatch windows: batch b in flight from first DISPATCH of b until first DISPATCH of b+1
disp = {}
rewards = {}
for line in open(orch, errors="replace"):
    m = re.search(r"DISPATCH prompt_id=\S+ prompt_idx=\d+ batch_idx=(\d+)", line)
    if m:
        b = int(m.group(1)); t = ts(line)
        if t and b not in disp: disp[b] = t
    m = re.search(r"METRIC step=(\d+) rewards/train/mean=([0-9.]+)", line)
    if m: rewards[int(m.group(1))] = float(m.group(2))
bounds = sorted(disp.items())
def batch_of(t):
    b = None
    for k, t0 in bounds:
        if t >= t0: b = k
    return b
REAL = ("file_editor", "execute_bash", "search", "finish")
per = collections.defaultdict(lambda: collections.Counter())
for r in recs:
    if r.get("kind") != "model": continue
    b = batch_of(r["t"])
    c = per[b]; c["turns"] += 1
    if r["fn"] in REAL: c["valid"] += 1
    if r["parse"].get("raw_valid"): c["raw_valid"] += 1
    if r["parse"].get("format") == "tool_call_bad_json": c["bad_json"] += 1
    if r["fn"] == "file_editor": c["cmd_" + str(r["params"].get("command"))] += 1
    elif r["fn"]: c["fn_" + r["fn"]] += 1
# trajectories from the rollout log: per global_step status, patch bytes, reward
traj = collections.defaultdict(list)
for line in open(roll, errors="replace"):
    m = re.search(r"traj_id=(\S+), global_step=(\d+), status=(\S+), reward=([0-9.]+), steps=(\d+), total_time=([0-9.]+)s.*?completion_tokens=(\d+)", line)
    if m: traj[int(m.group(2))].append(dict(status=m.group(3), reward=float(m.group(4)), turns=int(m.group(5)), total=float(m.group(6)), tok=int(m.group(7))))
pb = collections.defaultdict(list)
for line in open(roll, errors="replace"):
    m = re.search(r"patch_bytes=(\d+)", line)
    if m:
        s = re.search(r"global_step=(\d+)", line)
        if s: pb[int(s.group(1))].append(int(m.group(1)))
print("step | model turns | well-formed (mapped) | raw-valid | bad_json | top commands | trajectories (SUCCEEDED/other) | turns mean | reward>0 | reward mean (orch) | patches != 1170 B")
for b in sorted(set(list(per.keys()) + list(traj.keys()))):
    c = per.get(b, collections.Counter()); tj = traj.get(b, [])
    top = ", ".join(f"{k[4:]}:{v}" for k, v in c.most_common() if k.startswith(("cmd_", "fn_")))[:80]
    n = c["turns"] or 1
    print(f"{b} | {c['turns']} | {c['valid']/n:.3f} | {c['raw_valid']/n:.3f} | {c['bad_json']} | {top} | {sum(1 for x in tj if x['status']=='SUCCEEDED')}/{sum(1 for x in tj if x['status']!='SUCCEEDED')} | {sum(x['turns'] for x in tj)/max(1,len(tj)):.1f} | {sum(1 for x in tj if x['reward']>0)} | {rewards.get(b,'-')} | {sum(1 for v in pb.get(b,[]) if v!=1170)}/{len(pb.get(b,[]))}")
