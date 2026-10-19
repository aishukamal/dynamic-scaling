#!/usr/bin/env python3
"""rl60_status.py [N]: per-step status for the long run from the orchestrator stream, controller log and sandbox_live.txt.
Columns: step, end, wall, train, wait share, live samplers (from the controller poll nearest the step end), reward/512, MA10,
nonzero-adv frac, grad norm, completions mean len (orchestrator metric, env+tool tokens incl.), turns, MAX_CONTEXT share,
gen tok/turn (rollout trajectory lines by global_step), sandbox creating/unschedulable at step end, controller decisions in the step."""
import re,sys,glob,json,collections,statistics as st,os,datetime
RUN=os.environ.get("RUN","rl60"); D=os.path.expanduser(f"~/workspaces/rl-pool-autoscaler-ops/tunix-recipe2/logs/{RUN}")
orch=(glob.glob(f"{D}/live/{RUN}-orch-*.log")+glob.glob(f"{D}/live/{RUN}-orch-*.try"))[0]
M=collections.defaultdict(dict); END={}
for l in open(orch,errors="replace"):
    m=re.search(r"^(\S+) .*METRIC step=(\d+) (\S+)=([0-9.e+-]+)",l)
    if m: M[int(m.group(2))][m.group(3)]=float(m.group(4)); END[int(m.group(2))]=m.group(1)[11:19]
T=collections.defaultdict(list)
for f in glob.glob(f"{D}/live/{RUN}-roll-*"):
    for l in open(f,errors="replace"):
        if "[DEBUG_INFERENCE][Trajectory]" not in l: continue
        d=dict(re.findall(r"(\w+)=([^,\s]+)",l))
        try: T[int(d["global_step"])].append((d["status"],float(d["reward"]),int(d["steps"]),int(d["completion_tokens"])))
        except Exception: pass
SB=[]
for l in open(f"{D}/sandbox_live.txt",errors="replace"):
    m=re.match(r"(\S+) (\{.*\})",l.strip())
    if m:
        try: SB.append((m.group(1)[11:19],eval(m.group(2),{},{})))
        except Exception: pass
def sb_at(t):
    b={}
    for ts,d in SB:
        if ts<=t: b=d
    return b
ctl=[]
cf=f"{D}/controller.jsonl"
if os.path.exists(cf):
    for l in open(cf,errors="replace"):
        try: r=json.loads(l)
        except Exception: continue
        ctl.append(r)
def live_at(t):
    best=None
    for r in ctl:
        ts=datetime.datetime.fromtimestamp(r["t"],datetime.timezone.utc).strftime("%H:%M:%S")
        if ts<=t and r.get("kind")=="poll": best=r
    if not best: return "-"
    w=best.get("status",{}).get("workers",{}); return str(sum(1 for v in w.values() if str(v).startswith("active")))
acts=collections.defaultdict(list)
for r in ctl:
    d=r.get("decision") or {}
    if d.get("action") in ("scale_out","scale_in"):
        ts=datetime.datetime.fromtimestamp(r["t"],datetime.timezone.utc).strftime("%H:%M:%S")
        s=r.get("status",{}).get("last_completed_step"); acts[s].append(f"{ts} {d['action']}->{d.get('desired')} ({d.get('reason','')[:60]})")
rewards=[]; out=[]
hdr="| step | end | wall s | train s | wait share | live | reward/512 | MA10 | nonzero-adv | grad norm | compl mean len | turns | MAX_CTX | gen tok/turn | sbx creating | unsched | decisions |"
out.append(hdr); out.append("|"+"---|"*17)
for s in sorted(M):
    m=M[s]; r=m.get("rewards/train/sum",0.0); rewards.append(r); ma=st.mean(rewards[-10:])
    t=T.get(s,[]); sb=sb_at(END[s])
    mc=f"{sum(1 for x in t if x[0]=='MAX_CONTEXT_LIMIT_REACHED')/len(t):.2f}" if t else "-"
    tm=f"{st.mean(x[2] for x in t):.1f}" if t else "-"; gt=f"{st.mean(x[3]/x[2] for x in t if x[2]>0):.0f}" if t else "-"
    g=m.get("actor/train/grad_norm"); 
    out.append(f"| {s} | {END[s]} | {m.get('orchestrator/train/step_time_sec',0):.0f} | {m.get('orchestrator/train/policy_training_time',0):.0f} | {m.get('orchestrator/train/trainer_wait_ratio', (m.get('orchestrator/train/exposed_generation_time',0)/max(1,m.get('orchestrator/train/step_time_sec',1)))):.2f} | {live_at(END[s])} | {r:.0f} | {ma:.1f} | {m.get('rewards/train/advantage/nonzero_frac',0):.3f} | {'-' if g is None else f'{g:.3f}'} | {m.get('rollout/train/completions/mean_length',0):.0f} | {tm} | {mc} | {gt} | {sb.get('creating','-')} | {sb.get('unschedulable',0)} | {'; '.join(acts.get(s,[]))} |")
print("\n".join(out))
if len(rewards)>=10: print(f"\nMA(1-10)={st.mean(rewards[1:11]) if len(rewards)>=11 else st.mean(rewards[:10]):.2f}", end="")
if len(rewards)>=61: print(f"  MA(51-60)={st.mean(rewards[51:61]):.2f}", end="")
print(f"  last10={st.mean(rewards[-10:]):.2f} n={len(rewards)} total={sum(rewards):.0f}")
