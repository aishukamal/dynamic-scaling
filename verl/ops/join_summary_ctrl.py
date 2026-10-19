#!/usr/bin/env python3
"""join_summary_ctrl.py <decisions jsonl> <pool jsonl> : per controller scale-out, the product-path breakdown:
decision -> RayCluster patch -> pod Running -> Ray node free -> POST /scale -> spawn_start (next sync) -> init_standalone
(placement group + worker group + vLLM-TPU engine init + compile) -> join sync -> LB add -> first token; decision-to-serving total."""
import json, sys
from datetime import datetime, timezone
def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
dec = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
pool = [json.loads(l) for l in open(sys.argv[2]) if l.startswith("{")]
acts = [r["actuator"] for r in dec if r.get("kind") == "actuator" and isinstance(r.get("actuator"), dict) and r["actuator"].get("kind") == "out"]
spawns = {}
for r in pool:
    if r["kind"] in ("spawn_start", "spawn_ready", "spawn_joined", "join_on_ready", "join_on_ready_error", "join_on_ready_skipped"):
        spawns.setdefault(r["rank"], {})[r["kind"]] = r
def d(a, b): return round(b - a, 1) if a is not None and b is not None else None
seen = set()
for a in acts:
    key = (a.get("desired"), a.get("t_decision"))
    if key in seen: continue
    seen.add(key)
    t_post = a.get("t_post_scale")
    cand = sorted([s for s in spawns.values() if "spawn_start" in s and t_post and ts(s["spawn_start"]["t"]) >= t_post - 1], key=lambda s: ts(s["spawn_start"]["t"]))
    s = cand[0] if cand else {}
    st, rd, jn = s.get("spawn_start"), s.get("spawn_ready"), s.get("spawn_joined") or s.get("join_on_ready")
    out = {"desired": a.get("desired"), "decision_utc": datetime.fromtimestamp(a["t_decision"], timezone.utc).strftime("%H:%M:%S"),
           "pod_needed": a.get("pod_needed"), "decision_to_patch_s": d(a["t_decision"], a.get("t_patch")),
           "patch_to_pod_running_s": d(a.get("t_patch"), a.get("t_pod_running")), "pod_running_to_node_free_s": d(a.get("t_pod_running"), a.get("t_node_ready")),
           "node_free_to_post_s": d(a.get("t_node_ready"), t_post), "post_to_spawn_start_s": d(t_post, ts(st["t"]) if st else None),
           "init_standalone_s": rd and round(rd["init_standalone_s"], 1), "ready_to_lb_add_s": jn and round(jn.get("ready_to_lb_add_s") or 0, 1),
           "first_token_s": jn and jn.get("first_token_s") and round(jn["first_token_s"], 2),
           "decision_to_serving_s": d(a["t_decision"], ts(jn["t"]) if jn else None), "rank": st["rank"] if st else None, "error": a.get("error")}
    if jn and jn.get("kind") == "join_on_ready":
        out.update({"join_path": "on_ready", "attempts": rd and rd.get("attempts"), "pod_ray_s": jn.get("pod_ray_s") and round(jn["pod_ray_s"], 1),
                    "engine_init_s": jn.get("engine_init_s") and round(jn["engine_init_s"], 1),
                    "wait_clean_weights_s": round(jn.get("wait_clean_weights_s") or 0, 1), "actor_busy_at_request": jn.get("actor_busy_at_request"),
                    "joiner_sync_s": round(jn.get("joiner_sync_s") or 0, 1), "sync_phases": {k: round(v, 1) for k, v in (jn.get("phases") or {}).items()},
                    "lb_add_s": round(jn.get("lb_add_s") or 0, 2), "version": jn.get("version"), "ready_to_serving_s": round(jn.get("ready_to_serving_s") or 0, 1)})
    elif jn:
        out["join_path"] = "next_sync"
    print(json.dumps(out))
