#!/usr/bin/env bash
# stop_after.sh <job> <stage pod> <pool jsonl> <N> : polls the hook JSONL; when N step records exist, stops the Ray job (static rung complete)
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=60s"
JOB=$1; SPOD=$2; POOL=$3; N=$4; D="$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3"
HEAD=$($K get pods -l ray.io/node-type=head,ray.io/cluster=${CLUSTER:-ray-tpum3} -o jsonpath='{.items[0].metadata.name}')
# the driver (and so the pool JSONL, a per-host hostPath) may land on any TPU-M3 host: find the stage pod that has the file
while true; do
  for P in "$SPOD" tpum3-stage-cb1q tpum3-stage-jpj1 tpum3-stage-0mw2 tpum3-stage-588s; do
    if $K exec "$P" -- test -f "$POOL" >/dev/null 2>&1; then SPOD=$P; break; fi
  done
  $K cp "$SPOD:$POOL" "$D/results/pool-$JOB.partial.jsonl" >/dev/null 2>&1
  C=$(grep -c '"kind": "step"' "$D/results/pool-$JOB.partial.jsonl" 2>/dev/null || echo 0)
  echo "$(date -u +%FT%TZ) steps=$C"
  if [ "${C:-0}" -ge "$N" ]; then echo "$(date -u +%FT%TZ) $N versions logged: stopping $JOB"; $K exec "$HEAD" -- ray job stop "$JOB" 2>&1 | tail -1; break; fi
  sleep 30
done
