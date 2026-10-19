#!/usr/bin/env bash
# submit.sh <submission-id> <runtime_env json file in this dir> [ENV=VAL ...] : runs run_tpum2_disagg.sh as a Ray job on ray-tpum3
set -euo pipefail
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX"
SUB=$1; RT=$2; shift 2; ENVS="$*"
HEAD=$($K get pods -l ray.io/node-type=head,ray.io/cluster=${CLUSTER:-ray-tpum3} -o jsonpath='{.items[0].metadata.name}')
$K cp "$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3/$RT" "$HEAD:/home/ray/$RT"
$K exec "$HEAD" -- bash -c "cd /verl && ray job submit --address http://localhost:8265 --no-wait --runtime-env-json \"\$(cat /home/ray/$RT)\" --submission-id $SUB -- bash -c 'cd /verl && $ENVS bash /data/tpu-m3/${SCRIPT:-run_tpum3_swe_disagg.sh}' 2>&1 | grep -iE 'submitted|error' | head -3"
date -u +%FT%TZ
