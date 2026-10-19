#!/usr/bin/env python3
"""analyze_pool.py <job.log> [--levels] : per-step rows from [POOL] step records (n_active, step wall, gen wait, sync wall, reward) and
per-level means (steps grouped by n_active, first 2 steps after each change excluded), steps per sampler-chip-hour = 3600 / (step_s * n_active)."""
import json, re, sys, statistics as st
log = re.sub(r"\x1b\[[0-9;]*m", "", open(sys.argv[1], errors="replace").read())
rows = []
def _recs(text):
    for line in text.splitlines():
        if line.startswith("{"):
            try: yield json.loads(line)
            except json.JSONDecodeError: pass
            continue
        if "[POOL] {" not in line: continue
        try: yield json.loads(line[line.index("[POOL] {") + 7 : line.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError): continue
for r in _recs(log):
    if r.get("kind") != "step": continue
    mt = r["metrics"]
    rows.append(dict(step=r["step"], n=r["n_active"], t=r["t"][11:19], step_s=mt.get("timing_s/step"), gen=mt.get("timing_s/gen"),
                     upd_actor=mt.get("timing_s/update_actor"), upd_w=mt.get("timing_s/update_weights"), sync=mt.get("timing_s/tpu-sync/total_sync"),
                     p2p=mt.get("timing_s/tpu-sync/p2p_transfer"), h2d=mt.get("timing_s/tpu-sync/sampler_h2d"), tok=mt.get("perf/total_num_tokens"),
                     thr=mt.get("perf/throughput"), reward=mt.get("critic/score/mean"), rl=mt.get("response_length/mean")))
rows.sort(key=lambda r: r["step"])
print("step | t | n_active | step_s | gen_wait | upd_actor | upd_weights | sync | p2p | h2d | tokens | tok/s | reward | resp_len | steps/chip-h")
for r in rows:
    sch = 3600.0 / (r["step_s"] * r["n"]) if r["step_s"] and r["n"] else None
    f = lambda v, d=2: "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    print(f"{r['step']} | {r['t']} | {r['n']} | {f(r['step_s'])} | {f(r['gen'])} | {f(r['upd_actor'])} | {f(r['upd_w'])} | {f(r['sync'])} | {f(r['p2p'],3)} | {f(r['h2d'],3)} | {f(r['tok'],0)} | {f(r['thr'],0)} | {f(r['reward'],3)} | {f(r['rl'],0)} | {f(sch)}")
if "--levels" in sys.argv:
    print("\n== levels (grouped by n_active; first 2 steps after a change excluded) ==")
    print("n_active | steps used | mean step_s | mean gen_wait | idle ratio | mean upd_actor | mean sync | tok/s mean | steps/sampler-chip-h | reward mean")
    groups, prev, run = {}, None, 0
    for r in rows:
        if r["n"] != prev: run = 0; prev = r["n"]
        run += 1
        if run > 2: groups.setdefault(r["n"], []).append(r)
    for n in sorted(groups):
        g = groups[n]; ms = st.mean(x["step_s"] for x in g); mg = st.mean(x["gen"] for x in g)
        print(f"{n} | {len(g)} ({g[0]['step']}-{g[-1]['step']}) | {ms:.2f} | {mg:.2f} | {mg/ms:.2f} | {st.mean(x['upd_actor'] for x in g):.2f} | "
              f"{st.mean(x['sync'] for x in g if x['sync'] is not None):.2f} | {st.mean(x['thr'] for x in g):.0f} | {3600/(ms*n):.1f} | {st.mean(x['reward'] for x in g):.3f}")
