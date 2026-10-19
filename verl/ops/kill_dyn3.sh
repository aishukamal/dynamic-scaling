#!/usr/bin/env bash
# kill_dyn3.sh <rank> [chip=1] : failure act for the dyn3 run. Snapshots the pool /status (per-rank LB in-flight) to results/kill-<job>-pre.json,
# maps the rank to its pod/host from the spawn_ready address, SIGKILLs the vLLM worker holding /dev/vfio/<chip> on that host (kill_replica.sh),
# snapshots /status again. CLUSTER env (default ray-tpum3), JOB env (default tpum3-dyn3).
set -euo pipefail
R=$1; CHIP=${2:-1}; JOB=${JOB:-tpum3-dyn3}; export CLUSTER=${CLUSTER:-ray-tpum3}
D="$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3"; cd "$D"
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=60s"
HEAD=$($K get pods -l ray.io/node-type=head,ray.io/cluster=$CLUSTER -o jsonpath='{.items[0].metadata.name}')
IP=$(python3 -c "
import json,sys
ip=None
for l in open('results/pool-$JOB.partial.jsonl'):
    d=json.loads(l)
    if d['kind']=='init': ip={int(k):v.split(':')[0] for k,v in d['addresses'].items()}.get($R, ip)
    if d['kind']=='spawn_ready' and d['rank']==$R: ip=d['address'].split(':')[0]
print(ip or '')")
[ -n "$IP" ] || { echo "no address for rank $R"; exit 1; }
POD=$($K get pods -l ray.io/cluster=$CLUSTER,ray.io/group=tpu-group-1 -o wide --no-headers | awk -v ip="$IP" '$6==ip {print $1, $7}')
HOST=$(echo "$POD" | awk '{print $2}')
echo "rank $R ip $IP pod/host: $POD (short $HOST)"
$K exec "$HEAD" -- python3 /home/ray/rlpool/ctl.py 127.0.0.1 status > "results/kill-$JOB-pre.json"; python3 -c "import json; d=json.load(open('results/kill-$JOB-pre.json')); print('PRE active', d.get('active'), 'inflight', d.get('inflight'), 'versions', d.get('versions'))"
./kill_replica.sh "$CHIP" "$HOST" | tee -a "results/kill-$JOB.log"
sleep 2
$K exec "$HEAD" -- python3 /home/ray/rlpool/ctl.py 127.0.0.1 status > "results/kill-$JOB-post.json"; python3 -c "import json; d=json.load(open('results/kill-$JOB-post.json')); print('POST active', d.get('active'), 'inflight', d.get('inflight'))"
date -u +%FT%T.%3NZ
