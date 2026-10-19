#!/usr/bin/env python3
"""join_summary.py <drive log> <pool jsonl> : per cold join, the product-path breakdown:
  patch -> pod Running -> Ray node free (from the drive log's PATCH / POD RUNNING / RAY NODE FREE / POSTED lines),
  POST /scale -> spawn_start (hook, at the next sync) -> init_standalone (placement group + workers + vLLM-TPU engine init + compile)
  -> join sync -> LB add -> first token (hook spawn_* records), and the decision-to-serving total = PATCH -> LB add + first token."""
import json, re, sys
from datetime import datetime, timezone
def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
drive = open(sys.argv[1]).read().splitlines()
pool = [json.loads(l) for l in open(sys.argv[2]) if l.startswith("{")]
patches = []
cur = None
for l in drive:
    m = re.match(r"(\S+) PATCH replicas=(\d+)", l)
    if m: cur = {"t_patch": ts(m.group(1)), "replicas": int(m.group(2))}; patches.append(cur); continue
    if cur is None: continue
    m = re.match(r"(\S+) POD RUNNING", l)
    if m: cur["t_pod"] = ts(m.group(1))
    m = re.match(r"(\S+) RAY NODE FREE", l)
    if m: cur["t_node"] = ts(m.group(1))
    m = re.match(r"(\S+) POSTED /scale", l)
    if m: cur["t_post"] = ts(m.group(1))
spawns = {}
for r in pool:
    if r["kind"] in ("spawn_start", "spawn_ready", "spawn_joined"):
        spawns.setdefault(r["rank"], {})[r["kind"]] = r
for p in patches:
    # the spawn whose spawn_start follows this POST
    cand = [s for s in spawns.values() if "spawn_start" in s and ts(s["spawn_start"]["t"]) >= p.get("t_post", p["t_patch"])]
    cand.sort(key=lambda s: ts(s["spawn_start"]["t"]))
    s = cand[0] if cand else {}
    st, rd, jn = s.get("spawn_start"), s.get("spawn_ready"), s.get("spawn_joined")
    def d(a, b): return round(b - a, 1) if a is not None and b is not None else None
    out = {"replicas": p["replicas"], "patch_utc": datetime.fromtimestamp(p["t_patch"], timezone.utc).strftime("%H:%M:%S"),
           "patch_to_pod_running_s": d(p["t_patch"], p.get("t_pod")), "pod_to_ray_node_free_s": d(p.get("t_pod"), p.get("t_node")),
           "node_free_to_post_s": d(p.get("t_node"), p.get("t_post")),
           "post_to_spawn_start_s": d(p.get("t_post"), ts(st["t"]) if st else None),
           "init_standalone_s": rd and round(rd["init_standalone_s"], 1),
           "ready_to_lb_add_s": jn and round(jn.get("ready_to_lb_add_s") or 0, 1),
           "first_token_s": jn and jn.get("first_token_s") and round(jn["first_token_s"], 2),
           "patch_to_serving_s": d(p["t_patch"], ts(jn["t"]) if jn else None),
           "rank": st["rank"] if st else None}
    print(json.dumps(out))
