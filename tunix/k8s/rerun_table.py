#!/usr/bin/env python3
"""rerun_table.py <run:label:chips> ... : comparison table for the serial static reruns (steps 1-7 means) + per-step lines."""
import csv,sys,statistics as st,re
out=["| arm | run | hosts | chips held | steps 1-7 mean s | median | min / max | train s | exposed gen s | chip-s per step | steps per chip-h | reward sum (steps 0-7, /4096) | rewarded lines / lines | MAX_CONTEXT | turns | gen tok/turn | well-formed | sbx creating at step ends (min-max) | unschedulable max |","|"+"---|"*19]
lines=[]
for arg in sys.argv[1:]:
    run,label,chips,hosts=arg.split("|"); chips=int(chips)
    rows=list(csv.DictReader(open(f"logs/{run}/steps.csv"))); s17=[r for r in rows if 1<=int(r["step"])<=7]
    if len(s17)<7: note=f" ({len(s17)} of 7 steps)"
    else: note=""
    w=[float(r["wall"]) for r in s17]; tr=st.mean(float(r["train"]) for r in s17); eg=st.mean(float(r["exposed_gen"]) for r in s17)
    chip=st.mean(w)*chips; rew=sum(float(r["reward_sum"] or 0) for r in rows)
    t=open(f"logs/{run}/traj_lines.txt").read(); rl=re.search(r"reward>0: (\d+) of (\d+)",t); mc=re.search(r"'MAX_CONTEXT_LIMIT_REACHED': (\d+)",t); sc=re.search(r"'SUCCEEDED': (\d+)",t); tm=re.search(r"turns: mean ([0-9.]+)",t); gt=re.search(r"generated tokens per turn: mean (\d+)",t)
    d=open(f"logs/{run}/dump_summary.txt").read(); wf=re.search(r"well-formed \(raw or mapped\) ([0-9.]+)",d)
    cr=[int(r["sbx_creating_at_end"]) for r in rows if r["sbx_creating_at_end"] not in ("",None)]; un=[int(float(r["sbx_unschedulable_at_end"] or 0)) for r in rows]
    mcs=int(mc.group(1))/(int(mc.group(1))+int(sc.group(1)))
    out.append(f"| {label} | {run}{note} | {hosts} | {chips} | {st.mean(w):.0f} | {st.median(w):.0f} | {min(w):.0f} / {max(w):.0f} | {tr:.0f} | {eg:.0f} | {chip:.0f} | {3600/chip:.2f} | {rew:.0f} | {rl.group(1)} / {rl.group(2)} | {mcs:.2f} | {tm.group(1)} | {gt.group(1)} | {wf.group(1) if wf else '-'} | {min(cr)}-{max(cr)} | {max(un)} |")
    lines.append(f"* {label} ({run}): wall " + " / ".join(f"{float(r['wall']):.0f}" for r in rows) + "; rewards " + " / ".join(f"{float(r['reward_sum'] or 0):.0f}" for r in rows) + "; grad norm " + " / ".join((f"{float(r['grad_norm']):.3f}" if r['grad_norm'] else "-") for r in rows) + "; creating at step end " + " / ".join(str(r['sbx_creating_at_end']) for r in rows))
print("\n".join(out)); print(); print("\n".join(lines))
