#!/usr/bin/env python3
"""recipe_csv.py <steps.csv from analyze_sg.py> <orch log> <out.csv> [analyze_dump table.txt]
Adds to the tg11d-layout CSV: reward_sum, reward_mean, nonzero_adv_groups_frac, grad_norm (actor, this step's update),
policy_training_time, prompts_mean_len, completions_mean_len, and tool-call validity per batch window (well_formed, raw_valid,
model_turns) from the analyze_dump table when given."""
import csv,re,sys,collections
steps_csv,orch,out=sys.argv[1],sys.argv[2],sys.argv[3]; dump_tbl=sys.argv[4] if len(sys.argv)>4 else None
M=collections.defaultdict(dict)
for l in open(orch,errors="replace"):
    m=re.search(r"METRIC step=(\d+) (\S+)=([0-9.e+-]+)",l)
    if m: M[int(m.group(1))][m.group(2)]=float(m.group(3))
V={}
if dump_tbl:
    for l in open(dump_tbl):
        c=[x.strip() for x in l.split("|")]
        if len(c)>4 and c[0].isdigit(): V[int(c[0])]={"model_turns":c[1],"well_formed":c[2],"raw_valid":c[3]}
rows=list(csv.DictReader(open(steps_csv)))
extra=["reward_sum","reward_mean","nonzero_adv_groups_frac","grad_norm","policy_training_time","prompts_mean_len","completions_mean_len","model_turns","well_formed","raw_valid"]
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())+extra); w.writeheader()
    for r in rows:
        s=int(r["step"]); m=M.get(s,{}); v=V.get(s,{})
        r.update({"reward_sum":m.get("rewards/train/sum",""),"reward_mean":m.get("rewards/train/mean",""),
                  "nonzero_adv_groups_frac":m.get("rewards/train/advantage/nonzero_frac",""),"grad_norm":m.get("actor/train/grad_norm",""),
                  "policy_training_time":m.get("orchestrator/train/policy_training_time",""),"prompts_mean_len":m.get("rollout/train/prompts/mean_length",""),
                  "completions_mean_len":m.get("rollout/train/completions/mean_length",""),
                  "model_turns":v.get("model_turns",""),"well_formed":v.get("well_formed",""),"raw_valid":v.get("raw_valid","")})
        w.writerow(r)
print("wrote",out,len(rows),"rows")
