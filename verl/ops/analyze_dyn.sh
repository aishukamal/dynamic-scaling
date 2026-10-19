#!/usr/bin/env bash
# analyze_dyn.sh <job> <pool host short> [PDT start "YYYY-MM-DD HH:MM:SS"] [PDT end] : (worker logs hold every job of the Ray session on a host, so bound the window) final artefacts for a dynamic arm: pool JSONL + sandbox JSONL -> per-version CSVs (dyn + one-pager
# layouts), worker-log completion lines from all TPU-M3 hosts (un-deduplicated) -> per-version sandbox latencies and per-trajectory stats.
set -euo pipefail
JOB=$1; PH=$2; LO=${3:-2000-01-01 00:00:00}; HI=${4:-2999-01-01 00:00:00}; D="$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3"; cd "$D"
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=300s"
$K exec tpum3-stage-$PH -- cat /data/tpu-m3/pool-$JOB.jsonl > results/pool-$JOB.jsonl
TRAINER_CHIPS=8 python3 build_csv.py results/pool-$JOB.jsonl results/sandboxes-$JOB.jsonl results/${JOB#tpum3-}-versions.csv | tail -1
python3 onepager_csv.py results/${JOB#tpum3-}-versions.csv results/${JOB#tpum3-}-onepager.csv | tail -1
: > results/$JOB-worker-completions.txt
for H in $($K get pods -l app=tpum3-stage -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | sed 's/tpum3-stage-//'); do
  $K exec tpum3-stage-$H -- find /hosttmp/ray -name 'worker-*.err' -newer /dev/null -exec grep -ah "swe_agent .*reward=\|swe_agent rollout error\|pristine grading failed" {} + 2>/dev/null >> results/$JOB-worker-completions.txt || true
done
python3 - "$JOB" "$LO" "$HI" <<'PY2'
import re,sys
job,lo,hi=sys.argv[1:4]; t=re.compile(r':(2026-\d\d-\d\d \d\d:\d\d:\d\d),'); n=0
with open(f'results/{job}-worker-completions.txt', errors='replace') as f, open(f'results/{job}-completions.filtered.txt','w') as o:
    for l in f:
        m=t.search(l)
        if m and lo<=m.group(1)<=hi: o.write(l); n+=1
print('worker completion lines in window:', n)
PY2
python3 sbx_latency.py results/$JOB-completions.filtered.txt results/pool-$JOB.jsonl | tee results/${JOB#tpum3-}-sbx-latency.txt
python3 - "$JOB" <<'PY'
import re,sys,json,statistics as st
from collections import Counter
job=sys.argv[1]
pat=re.compile(r'swe_agent (\S+): reward=([0-9.]+) reason=(.*?) turns=(\d+) resp_tokens=(\d+) gen_s=([0-9.]+) tool_s=([0-9.]+)'); rep=re.compile(r'\[repeated (\d+)x')
rows=[]
for l in open(f'results/{job}-completions.filtered.txt', errors='replace'):
    m=pat.search(l)
    if not m: continue
    n=int(rep.search(l).group(1)) if rep.search(l) else 1
    rows+=[dict(reason=m.group(3).split(' stats=')[0].split(' (')[0],reward=float(m.group(2)),turns=int(m.group(4)),tok=int(m.group(5)),gen=float(m.group(6)),tool=float(m.group(7)))]*n
print(job,'trajectories',len(rows),' '.join(f"{k} {st.mean([r[k] for r in rows]):.1f}/{st.median([r[k] for r in rows]):.1f}" for k in ('gen','tool','turns','tok','reward')))
c=Counter(r['reason'] for r in rows); print(' reasons', [(k,v,f"{100*v/len(rows):.1f}%") for k,v in c.most_common(8)])
PY
