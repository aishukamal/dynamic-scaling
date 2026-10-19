#!/usr/bin/env python3
"""reward_stats.py <filtered job log> : per-version table for a reward run from the loop's completion lines
(swe_agent <id>: reward=.. reason=.. turns=.. stats={...}) and the trainer's step metric lines. Versions are delimited by the
trainer's 'timing_s/step' log (metrics printed at the end of each step); a completion belongs to the version in whose window
it completed. Groups = same instance_id within a version (n=4); nonzero advantage = rewards differ within the group."""
import re, sys, json, collections
ansi = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"(?:INFO|WARNING|ERROR):(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
COMP = re.compile(r"swe_agent (\S+): reward=([0-9.]+) reason=(\S+) turns=(\d+) resp_tokens=(\d+) gen_s=([0-9.]+) tool_s=([0-9.]+) stats=(\{.*\})")
STEP = re.compile(r"step:(\d+) - (.*)$")
comps, steps = [], []
seen = set()
for raw in open(sys.argv[1], errors="replace"):
    l = ansi.sub("", raw.rstrip("\n"))
    m = COMP.search(l)
    if m:
        t = TS.search(l)
        key = (t.group(1) if t else None, m.group(1), m.group(3), m.group(4), m.group(5))
        if key in seen:  # Ray prints a worker line twice (with and without the ip prefix)
            continue
        seen.add(key)
        comps.append({"t": t.group(1) if t else None, "id": m.group(1), "reward": float(m.group(2)), "reason": m.group(3),
                      "turns": int(m.group(4)), "resp": int(m.group(5)), "gen_s": float(m.group(6)), "tool_s": float(m.group(7)),
                      "stats": json.loads(m.group(8)), "line": len(comps) + len(steps)})
        continue
    m = STEP.search(l)
    if m and "timing_s/step" in l:
        kv = {}
        for part in m.group(2).split(" - "):
            if ":" in part:
                k, v = part.rsplit(":", 1)
                try: kv[k.strip()] = float(v)
                except ValueError: pass
        steps.append({"step": int(m.group(1)), "metrics": kv, "line": len(comps) + len(steps)})
print(f"completions {len(comps)}, step logs {len(steps)}")
# assign completions to versions by order: a completion before step k's log line belongs to version k
rows = []
ci = 0
for s in steps:
    mine = [c for c in comps[ci:] if c["line"] < s["line"]]
    ci += len(mine)
    groups = collections.defaultdict(list)
    for c in mine: groups[c["id"]].append(c["reward"])
    nz = sum(1 for g in groups.values() if len(set(g)) > 1)
    st = collections.Counter()
    for c in mine:
        for k, v in c["stats"].items(): st[k] += v
    m = s["metrics"]
    rows.append({"version": s["step"], "n": len(mine), "reward_mean": round(sum(c["reward"] for c in mine) / max(1, len(mine)), 4),
                 "frac_nonzero": round(sum(1 for c in mine if c["reward"] > 0) / max(1, len(mine)), 3),
                 "groups": len(groups), "groups_nonzero_adv": nz,
                 "wellformed_turn_frac": round(st["wellformed"] / max(1, st["turns"]), 3), "rc0_frac": round(st["rc0"] / max(1, st["executed"]), 3),
                 "exec_per_traj": round(st["executed"] / max(1, len(mine)), 2), "submit_rejected": st["submit_rejected"],
                 "patch_nonempty": st["patch_nonempty"], "turns_mean": round(sum(c["turns"] for c in mine) / max(1, len(mine)), 2),
                 "reasons": dict(collections.Counter(c["reason"] for c in mine).most_common(4)),
                 "score_mean(trainer)": m.get("critic/score/mean"), "grad_norm": m.get("actor/grad_norm"), "step_s": m.get("timing_s/step"),
                 "gen_wait_s": m.get("timing_s/gen"), "update_actor_s": m.get("timing_s/update_actor"), "sync_s": m.get("timing_s/update_weights"),
                 "resp_len": m.get("response_length/mean"), "lr": m.get("actor/lr")})
left = comps[ci:]
for r in rows: print(json.dumps(r))
if left:
    st = collections.Counter(); groups = collections.defaultdict(list)
    for c in left:
        for k, v in c["stats"].items(): st[k] += v
        groups[c["id"]].append(c["reward"])
    print(json.dumps({"version": "in-progress", "n": len(left), "reward_mean": round(sum(c["reward"] for c in left) / len(left), 4),
                      "frac_nonzero": round(sum(1 for c in left if c["reward"] > 0) / len(left), 3), "groups": len(groups),
                      "groups_nonzero_adv": sum(1 for g in groups.values() if len(set(g)) > 1),
                      "wellformed_turn_frac": round(st["wellformed"] / max(1, st["turns"]), 3), "rc0_frac": round(st["rc0"] / max(1, st["executed"]), 3),
                      "exec_per_traj": round(st["executed"] / len(left), 2), "submit_rejected": st["submit_rejected"], "patch_nonempty": st["patch_nonempty"],
                      "turns_mean": round(sum(c["turns"] for c in left) / len(left), 2),
                      "reasons": dict(collections.Counter(c["reason"] for c in left).most_common(5))}))
