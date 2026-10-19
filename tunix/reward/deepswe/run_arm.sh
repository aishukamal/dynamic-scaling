#!/bin/bash
# run_arm.sh <run-name> <rollout-host-list>; config via env. Renders, applies, follows logs, waits for the orchestrator pod, collects, tears down.
set -uo pipefail
RUN=$1; HOSTS=$2; shift 2
OPS=~/workspaces/rl-pool-autoscaler-ops/tunix-reward; K=$OPS/k8s/kctl.sh; cd $OPS/deepswe
export TUNIX_IMAGE=${TUNIX_IMAGE:-us-west1-docker.pkg.dev/aishuk-test/tpu-poc-repo/tunix-poc:mlperf-65d021b-deepswe-raiden-20261015035906}
export JOB_PREFIX=${JOB_PREFIX:-tr} TRAINER_NODE=${TRAINER_NODE:-gke-tpu-dac1e83d-8f4h} ROLLOUT_NODES="$HOSTS"
mkdir -p runs $OPS/logs/$RUN
./render.sh runs/$RUN > runs/$RUN.render.log 2>&1 || { cat runs/$RUN.render.log; exit 1; }
echo "$(date -u +%FT%TZ) $RUN apply hosts=[$HOSTS] bs=${BATCH_SIZE:-1} n=${NUM_GENERATIONS:-4} steps=${MAX_STEPS:-10} seqs=${VLLM_MAX_NUM_SEQS:-8} conc=${ROLLOUT_MAX_CONCURRENCY:-64} extra_roll=[${ROLLOUT_EXTRA_ENV:-}]" | tee -a $OPS/logs/actions.log
./apply_run.sh runs/$RUN 2>&1 | tee -a $OPS/logs/actions.log
./follow_logs.sh $RUN $JOB_PREFIX > $OPS/logs/$RUN/follow.log 2>&1 & FOLLOW_PID=$!
T0=$(date +%s)
while true; do
  sleep 30
  ph=$($K get pods -n default -o jsonpath='{range .items[?(@.metadata.labels.jobset\.sigs\.k8s\.io/jobset-name=="'$JOB_PREFIX'-orch")]}{.status.phase}{end}' 2>/dev/null)
  [ "$ph" = "Succeeded" ] || [ "$ph" = "Failed" ] && break
  [ $(( $(date +%s) - T0 )) -gt ${ARM_TIMEOUT_S:-5400} ] && { echo "TIMEOUT"; break; }
done
echo "$(date -u +%FT%TZ) $RUN orchestrator phase=$ph after $(( $(date +%s) - T0 )) s" | tee -a $OPS/logs/actions.log
sleep 15; touch $OPS/logs/$RUN/live/.stop; sleep 12; kill $FOLLOW_PID 2>/dev/null; pkill -f "logs -f ${JOB_PREFIX}-" 2>/dev/null
./collect_logs.sh $RUN $JOB_PREFIX >/dev/null 2>&1
[ "${KEEP_RUNNING:-0}" = 1 ] || ./teardown_run.sh $JOB_PREFIX 2>&1 | tee -a $OPS/logs/actions.log
