#!/bin/bash
# render_run.sh <run-name> <prefix> <port-base> <exporter-port>; demand via BATCH_SIZE/NUM_GENERATIONS/MAX_STEPS env (defaults = te)
set -euo pipefail
RUN=$1; PFX=$2; PB=$3; EP=$4
OPS=~/workspaces/rl-pool-autoscaler-ops/tunix-recipe2; cd $OPS/k8s
export TUNIX_IMAGE=us-west1-docker.pkg.dev/aishuk-test/tpu-poc-repo/tunix-poc:mlperf-65d021b-deepswe-raiden-20261015035906
export JOB_PREFIX=$PFX ORCHESTRATOR_PORT=$PB ROLLOUT_PORT=$((PB+1)) TRAINER_PORT=$((PB+2)) EXPORTER_PORT=$EP
export TRAINER_NODE=${TRAINER_NODE:-gke-tpu-dac1e83d-gmhv} ROLLOUT_NODES="${ROLLOUT_NODES:-gke-tpu-dac1e83d-j020 gke-tpu-dac1e83d-smkc gke-tpu-dac1e83d-w9ln}" ROLLOUT_REPLICAS=${ROLLOUT_REPLICAS:-1}
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-1} TRAINER_MESH_TP=${TRAINER_MESH_TP:-4} ROLLOUT_MESH_FSDP=${ROLLOUT_MESH_FSDP:-1} ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-4}
export BATCH_SIZE=${BATCH_SIZE:-8} NUM_GENERATIONS=${NUM_GENERATIONS:-4} MAX_STEPS=${MAX_STEPS:-25}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64} ROLLOUT_MAX_CONCURRENCY=${ROLLOUT_MAX_CONCURRENCY:-64} MAX_CONCURRENCY=${MAX_CONCURRENCY:-${ROLLOUT_MAX_CONCURRENCY:-64}}
export SANDBOX_NODE_SELECTOR_KEY=${SANDBOX_NODE_SELECTOR_KEY:-sandbox.gke.io/runtime} SANDBOX_NODE_SELECTOR_VAL=${SANDBOX_NODE_SELECTOR_VAL:-gvisor}   # from tg7 on: both gVisor pools (coordinator 22:5xZ)
export MAX_STALENESS=${MAX_STALENESS:-}   # tg8+: 1 (user decision 23:25Z, mlperf_base.sh production value)
export SG_EXTRA_CP="cp /patch/swe_agent.py /app/examples/deepswe/swe_agent.py && cp /patch/r2_toolcall.py /app/examples/deepswe/r2_toolcall.py"
export ROLLOUT_EXTRA_ENV="TRAJ_DUMP_DIR=/tmp/tunix-traj/${RUN} TOOLCALL_MODE=${TOOLCALL_MODE:-} STRIP_THINK=${STRIP_THINK:-0} TOOLCALL_TOOL_ROLE=${TOOLCALL_TOOL_ROLE:-1}${ROLLOUT_EXTRA_ENV:+ ${ROLLOUT_EXTRA_ENV}}"
export ORCHESTRATOR_EXTRA_ENV="TOOLCALL_MODE=${TOOLCALL_MODE:-} STRIP_THINK=${STRIP_THINK:-0} TOOLCALL_TOOL_ROLE=${TOOLCALL_TOOL_ROLE:-1}${ORCHESTRATOR_EXTRA_ENV:+ ${ORCHESTRATOR_EXTRA_ENV}}"
export RUN_NAME=${RUN}-$(date -u +%H%M%S)
bash ~/workspaces/rl-pool-autoscaler-ops/tunix-slotgate/k8s/render-deepswe-sg.sh $OPS/runs/$RUN
grep -o "\-\-rollout_replicas=[0-9]*" $OPS/runs/$RUN/orchestrator.yaml
