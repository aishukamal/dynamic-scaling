#!/bin/bash
# Renders tunix deepswe_dist manifests for the eu cluster with DRY_RUN=true and post-processes them.
# Usage: render.sh <outdir>; env overrides are read from the caller. All kubectl calls go through the explicit-context shim.
set -euo pipefail
OUT=$1; mkdir -p "$OUT"
export PATH=~/workspaces/rl-pool-autoscaler-ops/tunix-deepswe/bin:$PATH
LAUNCHER=~/workspaces/tunix-poc/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh
PP=~/workspaces/rl-pool-autoscaler-ops/tunix-slotgate/k8s/postprocess-sg.py
export DRY_RUN=true
export TUNIX_IMAGE=${TUNIX_IMAGE:?}
export K8S_NAMESPACE=${K8S_NAMESPACE:-default}
export GANG_ID=""            # no gang label / no in-pod kubectl teardown
export JOB_PREFIX=${JOB_PREFIX:-te}
export ORCHESTRATOR_PORT=${ORCHESTRATOR_PORT:-20040} ROLLOUT_PORT=${ROLLOUT_PORT:-20041} TRAINER_PORT=${TRAINER_PORT:-20042}
export EXPORTER_PORT=${EXPORTER_PORT:-8789}
export CPU_NODEPOOL=${CPU_NODEPOOL:-pathways-head-jax}
export CPU_MEMORY=${CPU_MEMORY:-64G}
export SERVICE_ACCOUNT=default
export TRAINER_JOBSET_YAML=jobset.tpu.yaml ROLLOUT_JOBSET_YAML=jobset.tpu.yaml
export TRAINER_TPU_SLICE=tpuv6e:2x2 ROLLOUT_TPU_SLICE=tpuv6e:2x2
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-4} TRAINER_MESH_TP=${TRAINER_MESH_TP:-1}
export ROLLOUT_MESH_FSDP=${ROLLOUT_MESH_FSDP:-1} ROLLOUT_MESH_TP=${ROLLOUT_MESH_TP:-4}
HOST_ENV="TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_TOPOLOGY=2x2 TPU_ACCELERATOR_TYPE=v6e-4 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0"
SBX_ENV="SANDBOX_RUNTIME_CLASS=${SANDBOX_RUNTIME_CLASS:-gvisor}"
export TRAINER_EXTRA_ENV="${HOST_ENV}${TRAINER_EXTRA_ENV:+ ${TRAINER_EXTRA_ENV}}"
export ROLLOUT_EXTRA_ENV="${HOST_ENV} ${SBX_ENV}${ROLLOUT_EXTRA_ENV:+ ${ROLLOUT_EXTRA_ENV}}"
export ORCHESTRATOR_EXTRA_ENV="HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 ${SBX_ENV} TUNIX_EXPORTER_PORT=${EXPORTER_PORT} TUNIX_RUN_NAME=$(basename "$OUT")${ORCHESTRATOR_EXTRA_ENV:+ ${ORCHESTRATOR_EXTRA_ENV}}"
export MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B} MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B}
export MODEL_DIR=${MODEL_DIR:-/tmp/tunix-models/${MODEL_NAME}}
export TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_ID}}
export WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-raiden}
export USE_AGENT_SANDBOX=1
export SANDBOX_NAMESPACE=${SANDBOX_NAMESPACE:-agents-system}
export SANDBOX_NODE_SELECTOR_KEY=${SANDBOX_NODE_SELECTOR_KEY:-cloud.google.com/gke-nodepool}
export SANDBOX_NODE_SELECTOR_VAL=${SANDBOX_NODE_SELECTOR_VAL:-swe-gvisor-pool-eu}
export IMAGE_REWRITE_PREFIX=${IMAGE_REWRITE_PREFIX:-us-central1-docker.pkg.dev/aishuk-test/swe-mirror/namanjain12}
export DATASET_PATH=${DATASET_PATH:-/tmp/tunix-data/r2e-cal}
export RUN_NAME=${RUN_NAME:-$(basename "$OUT")-$(date -u +%H%M%S)}
export CHECKPOINT_ROOT_DIRECTORY=${CHECKPOINT_ROOT_DIRECTORY:-/tmp/tunix-ckpt/${RUN_NAME}}
export TRAINER_NODE=${TRAINER_NODE:?} ; ROLLOUT_NODES=(${ROLLOUT_NODES:?})
export ROLLOUT_REPLICAS=${ROLLOUT_REPLICAS:-${#ROLLOUT_NODES[@]}}
bash "$LAUNCHER" --dry-run --command orchestrator 2>"$OUT/render-orch.err" | python3 "$PP" orchestrator > "$OUT/orchestrator.yaml"
bash "$LAUNCHER" --dry-run --command trainer      2>"$OUT/render-trainer.err" | python3 "$PP" trainer "$TRAINER_NODE" > "$OUT/trainer.yaml"
i=0
for n in "${ROLLOUT_NODES[@]}"; do
  ROLLOUT_ID="${JOB_PREFIX}-roll-${i}" ROLLOUT_REPLICAS=1 bash "$LAUNCHER" --dry-run --command rollout 2>"$OUT/render-roll-$i.err" | python3 "$PP" rollout "$n" > "$OUT/rollout-$i.yaml"
  i=$((i+1))
done
cat > "$OUT/service.yaml" <<EOS
apiVersion: v1
kind: Service
metadata:
  name: ${JOB_PREFIX}-orch-metrics
  namespace: ${K8S_NAMESPACE}
  labels: {app: tunix-exporter, run-prefix: "${JOB_PREFIX}"}
spec:
  type: ClusterIP
  selector: {jobset.sigs.k8s.io/jobset-name: ${JOB_PREFIX}-orch}
  ports: [{name: metrics, port: ${EXPORTER_PORT}, targetPort: ${EXPORTER_PORT}}]
EOS
grep -H "kubernetes.io/hostname" "$OUT"/trainer.yaml "$OUT"/rollout-*.yaml
env | grep -E '^(EXPORTER_PORT|ORCHESTRATOR_PORT|RUN_NAME|MODEL_|BATCH_SIZE|NUM_GENERATIONS|MAX_STEPS|MAX_TURNS|MAX_RESPONSE|MAX_PROMPT|WEIGHT_SYNC|ROLLOUT_|TRAINER_|MINI_BATCH|TRAIN_MICRO|MAX_CONCURRENCY|MAX_WARMPOOL|VLLM_|SANDBOX_|IMAGE_REWRITE|DATASET_|JOB_PREFIX|CPU_|CHECKPOINT_|EPISODE_|STEP_TIMEOUT|REWARD_TIMEOUT|SEED|TUNIX_IMAGE)' | sort > "$OUT/render.env"
echo "rendered to $OUT"
