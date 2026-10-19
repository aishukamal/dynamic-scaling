#!/usr/bin/env python3
"""dump_summary.py <dump jsonl> <rollout log>: episode-level statistics from a TRAJ_DUMP_DIR dump + the rollout log."""
import json,collections,re,statistics as st,ast,sys
def D(x):
    if isinstance(x,dict): return x
    try: return ast.literal_eval(x) if x else {}
    except Exception:
        try: return json.loads(x)
        except Exception: return {}
recs=[]; _bad=0
for _l in open(sys.argv[1], errors="replace"):
    try: recs.append(json.loads(_l))
    except Exception: _bad+=1
if _bad: print(f"# skipped {_bad} undecodable dump lines")
m=[r for r in recs if r["kind"]=="model"]
eps=collections.defaultdict(list)
for r in recs: eps[r["agent"]].append(r)
P=[D(r.get("parse")) for r in m]
valid=sum(1 for p in P if p.get("raw_valid") or p.get("mapped")); raw=sum(1 for p in P if p.get("raw_valid"))
print(f"model turns {len(m)}; well-formed (raw or mapped) {valid/len(m):.3f}; raw-valid {raw/len(m):.3f}; episodes in dump {len(eps)}")
fn=collections.Counter(); cmd=collections.Counter()
for r in m:
    fn[r.get("fn")]+=1
    if r.get("fn")=="file_editor": cmd[D(r.get("params")).get("command")]+=1
print("tools:",fn.most_common()); print("file_editor commands:",cmd.most_common())
edit_eps=bash_eps=submit_eps=0
for v in eps.values():
    cmds=set(); fns=set()
    for r in v:
        if r["kind"]!="model": continue
        fns.add(r.get("fn"))
        if r.get("fn")=="file_editor": cmds.add(D(r.get("params")).get("command"))
    edit_eps+=bool(cmds & {"str_replace","create","insert"}); bash_eps+=("execute_bash" in fns); submit_eps+=("finish" in fns)
print(f"episodes with an edit (str_replace/create/insert) {edit_eps}/{len(eps)}; with execute_bash {bash_eps}; with finish {submit_eps}")
rl=[int(r["resp_len"]) for r in m]; print(f"completion tokens: mean {st.mean(rl):.0f} p50 {st.median(rl):.0f} max {max(rl)}")
cum=[sum(int(r["resp_len"]) for r in v if r["kind"]=="model") for v in eps.values()]
print(f"cumulative response tokens per episode: mean {st.mean(cum):.0f} p50 {st.median(cum):.0f} max {max(cum)}")
turns=[sum(1 for r in v if r["kind"]=="model") for v in eps.values()]
print("turns/episode: mean %.1f p50 %d max %d; hist %s"%(st.mean(turns),st.median(turns),max(turns),sorted(collections.Counter(turns).items())))
roll=open(sys.argv[2],errors="replace").read()
print("rollout log: MAX_CONTEXT", roll.count("MAX_CONTEXT_LIMIT_REACHED"), "| SUCCEEDED", len(re.findall(r"status=SUCCEEDED",roll)), "| empty patch", roll.count("Empty git patch extracted"), "| evaluated patches", roll.count("Evaluated agent patch"))
pb=collections.Counter(re.findall(r"patch_bytes=(\d+)",roll)); print("patch_bytes:",sorted(pb.items(),key=lambda x:-x[1]))
print("reward values in rollout log:",collections.Counter(re.findall(r"reward=([0-9.]+)",roll)))
# per-turn budget consumption from the engine's cur_tokens (generated + environment tokens after the first prompt), v3 dumps only
deltas=[]; finals=[]
for v in eps.values():
    v=sorted(v,key=lambda r: float(r["t"])); prev=None
    for r in v:
        if r["kind"]!="env" or r.get("cur_tokens") is None: continue
        c=int(r["cur_tokens"])
        if prev is not None: deltas.append(c-prev)
        prev=c
    if prev is not None: finals.append(prev)
if deltas:
    print(f"cur_tokens (engine budget): per-turn delta mean {st.mean(deltas):.0f} p50 {st.median(deltas):.0f} max {max(deltas)} over {len(deltas)} turns; final per episode mean {st.mean(finals):.0f} p50 {st.median(finals):.0f} max {max(finals)}")
# generated tokens per turn approximated from the orchestrator metric is not available per turn; characters per model turn above
