#!/usr/bin/env python3
"""long_analysis.py <dyn csv> <fixed csv> [skip=2] : long-arm comparison.
Prints: per-arm settled means (all / clean), matched-version means over the versions both arms completed (same version numbers),
drift = means over thirds of the run (idle, slot_util, step_s), chip-hours: sum(active x 4 x step_s)/3600 and versions per
sampler-chip-hour, the dynamic arm's active-count trajectory and events."""
import csv, sys, statistics as st
dyn = list(csv.DictReader(open(sys.argv[1]))); fix = list(csv.DictReader(open(sys.argv[2])))
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 2
def f(r, k): 
    try: return float(r[k])
    except (ValueError, KeyError, TypeError): return None
def summ(rows, label):
    rows = [r for r in rows if f(r, "step_s")]
    if not rows: print(label, "no rows"); return
    step = [f(r, "step_s") for r in rows]; wait = [f(r, "wait_s") for r in rows]
    act = [f(r, "active_replicas") for r in rows]
    chip_h = sum(a * 4 * s for a, s in zip(act, step)) / 3600
    util = [f(r, "slot_util") for r in rows if f(r, "slot_util") is not None]
    print(f"{label}: n={len(rows)} versions {rows[0]['version']}-{rows[-1]['version']} | step mean {st.mean(step):.1f} s | wait {st.mean(wait):.1f} | idle {sum(wait)/sum(step):.3f} | "
          f"slot_util {st.mean(util):.3f} | mean active {st.mean(act):.2f} | wall {sum(step)/3600:.2f} h | sampler chip-h {chip_h:.2f} | versions/wall-h {3600*len(rows)/sum(step):.2f} | versions/sampler-chip-h {len(rows)/chip_h:.3f} | reward mean {st.mean(f(r,'reward') or 0 for r in rows):.4f}")
print("== settled (after the first", skip, "versions)")
summ(dyn[skip:], "DYNAMIC all"); summ([r for r in dyn[skip:] if r.get("sandbox_confounded") in ("0", "")], "DYNAMIC clean")
summ(fix[skip:], "FIXED S=2 all"); summ([r for r in fix[skip:] if r.get("sandbox_confounded") in ("0", "")], "FIXED S=2 clean")
vd = {r["version"]: r for r in dyn}; vf = {r["version"]: r for r in fix}
common = [v for v in vd if v in vf and int(v) > skip]
print("== matched versions", common[0] if common else None, "-", common[-1] if common else None, f"(n={len(common)})")
summ([vd[v] for v in common], "DYNAMIC matched"); summ([vf[v] for v in common], "FIXED S=2 matched")
print("== drift (thirds of each arm's settled versions)")
for label, rows in (("DYNAMIC", dyn[skip:]), ("FIXED S=2", fix[skip:])):
    n = len(rows); 
    for i, part in enumerate((rows[: n // 3], rows[n // 3 : 2 * n // 3], rows[2 * n // 3 :])):
        if part: summ(part, f"  {label} third {i+1}")
print("== dynamic arm active-count trajectory:", [(r["version"], r["active_replicas"]) for i, r in enumerate(dyn) if i == 0 or r["active_replicas"] != dyn[i-1]["active_replicas"]])
print("== dynamic arm events:"); [print("  v" + r["version"], r["events"]) for r in dyn if r.get("events")]
