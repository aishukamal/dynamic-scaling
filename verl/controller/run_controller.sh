#!/usr/bin/env bash
# run_controller.sh <tag> [ENV=VAL ...] : starts the unchanged code/rl_pool_autoscaler/controller/main.py in the head pod (nohup) with
# the GPU-M2 Kubernetes actuator (K8S_ACTUATOR=1: RayCluster tpu-group-1 replicas +1 -> wait for a free rollout node in /status ->
# POST /scale; scale-in = POST /scale release=true -> patch replicas -1 with workersToDelete). JSONL at /home/ray/rlpool/decisions-<tag>.jsonl.
# Defaults below are the TPU-M3 settings; override on the command line (e.g. IDLE_HI=0.40).
set -euo pipefail
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=60s"
TAG=$1; shift
HEAD=$($K get pods -l ray.io/node-type=head,ray.io/cluster=${CLUSTER:-ray-tpum3} -o jsonpath='{.items[0].metadata.name}')
$K exec "$HEAD" -- env CONTROL_URL=http://${CONTROL_HOST:-127.0.0.1}:8787 DECISIONS_JSONL=/home/ray/rlpool/decisions-$TAG.jsonl PYTHONPATH=/home/ray/rlpool \
  K8S_ACTUATOR=1 RAYCLUSTER_NAME=${CLUSTER:-ray-tpum3} RAYCLUSTER_NS=default SAMPLER_GROUP=tpu-group-1 NODE_WAIT_TIMEOUT_S=900 RELEASE_ON_SCALE_IN=1 \
  MIN_STANDALONE=1 MAX_STANDALONE=3 DWELL_STEPS=2 POLL_INTERVAL_S=10 MIN_ACTION_INTERVAL_S=120 \
  UTIL_HI=0.75 UTIL_LO=0.40 FLOOR_MIN_GAIN=0.05 FLOOR_SKIP_VERSIONS=2 MQ_LO_FRAC=0.8 MQ_HI_FRAC=0.9 \
  "$@" python3 -m rl_pool_autoscaler.controller.main > "$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3/logs/controller-$TAG.out" 2>&1 &
sleep 8; echo "controller started ($TAG) at $(date -u +%FT%TZ) (kubectl exec session kept in the background; stdout = logs/controller-$TAG.out)"; tail -2 "$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3/logs/controller-$TAG.out" | cut -c1-300
