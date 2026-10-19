#!/bin/bash
# render_8chip.sh <run> <prefix> <port-base> <exporter-port> : rw4 settings at the recipe's 4096/4096 with the trainer on the
# 8-chip single host gke-tpu-2e88ab73-6nw8 (ct6e-standard-8t, topology 2x4). The launcher's yaml generator maps tpuv6e:2x4 to
# TWO 4-chip pods, so the trainer is rendered as a 2x2 single pod and edited here: topology label 2x4 (postprocess), tpu 4->8,
# host env 2,4,1 / 2x4 / v6e-8, RAIDEN_DEVICES_PER_HOST=8 (trainer only). Mesh via TRAINER_MESH_FSDP / TRAINER_MESH_TP env.
set -euo pipefail
RUN=$1; PFX=$2; PB=$3; EP=$4
OPS=~/workspaces/rl-pool-autoscaler-ops/tunix-recipe2
export TRAINER_NODE=${TRAINER_NODE:-gke-tpu-2e88ab73-6nw8} ROLLOUT_NODES="${ROLLOUT_NODES:-gke-tpu-7fb5c8f1-j6vt}" ROLLOUT_REPLICAS=${ROLLOUT_REPLICAS:-1}
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-8} TRAINER_MESH_TP=${TRAINER_MESH_TP:-1} ROLLOUT_MESH_FSDP=1 ROLLOUT_MESH_TP=4
export BATCH_SIZE=${BATCH_SIZE:-8} NUM_GENERATIONS=4 MAX_STEPS=${MAX_STEPS:-3}
export VLLM_MAX_NUM_SEQS=128 ROLLOUT_MAX_CONCURRENCY=128 MAX_CONCURRENCY=384
export MAX_STALENESS=1 IN_FLIGHT_WEIGHT_UPDATES=true
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export MODEL_NAME=Qwen3-4B MODEL_ID=Qwen/Qwen3-4B
export TOOLCALL_MODE=qwen_json STRIP_THINK=0 TOOLCALL_TOOL_ROLE=1
export SG_TPU_TOPOLOGY_LABEL_TRAINER=2x4 SG_TPU_TOPOLOGY_LABEL_ROLLOUT=2x2 SG_SPOT=1 SG_ORCH_CPU=${SG_ORCH_CPU:-6} CPU_MEMORY=${CPU_MEMORY:-24G}
${LEARNING_RATE:+export LEARNING_RATE=$LEARNING_RATE}
bash $OPS/k8s/render_run.sh "$RUN" "$PFX" "$PB" "$EP"
T=$OPS/runs/$RUN/trainer.yaml
python3 - "$T" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
for a,b,n in [('google.com/tpu: "4"','google.com/tpu: "8"',2),
              ('TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_TOPOLOGY=2x2 TPU_ACCELERATOR_TYPE=v6e-4','TPU_CHIPS_PER_HOST_BOUNDS=2,4,1 TPU_TOPOLOGY=2x4 TPU_ACCELERATOR_TYPE=v6e-8',1),
              ('RAIDEN_DEVICES_PER_HOST=4','RAIDEN_DEVICES_PER_HOST=8',1),
              ('cloud.google.com/gke-tpu-topology: 2x4\n              kubernetes.io/hostname: '+__import__('os').environ['TRAINER_NODE'],'cloud.google.com/gke-tpu-topology: 2x4\n              kubernetes.io/hostname: '+__import__('os').environ['TRAINER_NODE'],1)]:
    assert s.count(a)==n, (a, s.count(a)); s=s.replace(a,b)
open(p,"w").write(s); print("trainer.yaml edited for the 8-chip host")
PY
grep -n 'google.com/tpu: "8"\|TPU_ACCELERATOR_TYPE=v6e-8\|RAIDEN_DEVICES_PER_HOST=8\|mesh_fsdp=\|mesh_tp=' $T | cut -c1-160 | sed 's/  */ /g' | head
