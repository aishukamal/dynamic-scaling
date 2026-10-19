#!/usr/bin/env bash
# go_dynamic.sh <cluster> <job> <keep sampler host short> [MAX=3] [TRAIN_BS=128] : dynamic run on the EXISTING cluster.
# 1) scale the sampler group to ONE pod (the one on <keep host>; others listed in scaleStrategy.workersToDelete) and wait until Ray
#    shows 4 TPU in tpu-group-1; 2) submit the spawn-enabled job (ROLLOUT_NODES 1, initial_active 1, release on scale-in); 3) start
#    the unchanged controller in the head pod with the K8s actuator; 4) sandbox monitor under the job tag. Timestamps printed.
set -euo pipefail
C=$1; JOB=$2; KEEP=$3; MX=${4:-3}; BS=${5:-128}
D="$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3"; cd "$D"
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=120s"
IDX=$($K get raycluster $C -o json | python3 -c "import sys,json; g=json.load(sys.stdin)['spec']['workerGroupSpecs']; print([x['groupName'] for x in g].index('tpu-group-1'))")
PODS=$($K get pods -l ray.io/cluster=$C,ray.io/group=tpu-group-1 -o jsonpath='{range .items[*]}{.metadata.name} {.spec.nodeName}{"\n"}{end}')
DEL=$( (echo "$PODS" | grep -v "$KEEP" || true) | awk '{print $1}' | tr '\n' ' ')
DELJSON=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:]))" $DEL)
echo "$(date -u +%FT%TZ) scale-in to one sampler pod: keep $(echo "$PODS" | grep "$KEEP" | awk '{print $1}'), delete [$DEL]"
$K patch raycluster $C --type=json -p "[{\"op\":\"replace\",\"path\":\"/spec/workerGroupSpecs/$IDX/replicas\",\"value\":1},{\"op\":\"replace\",\"path\":\"/spec/workerGroupSpecs/$IDX/maxReplicas\",\"value\":$MX},{\"op\":\"add\",\"path\":\"/spec/workerGroupSpecs/$IDX/scaleStrategy\",\"value\":{\"workersToDelete\":$DELJSON}}]"
HEAD=$($K get pods -l ray.io/node-type=head,ray.io/cluster=$C -o jsonpath='{.items[0].metadata.name}')
for i in $(seq 1 60); do R=$($K get pods -l ray.io/cluster=$C,ray.io/group=tpu-group-1 --no-headers 2>/dev/null | wc -l | tr -d ' '); T=$($K exec "$HEAD" -- ray status 2>/dev/null | grep -E "^ [0-9.]+/[0-9.]+ tpu-group-1" | sed -E 's#.*/([0-9.]+) tpu-group-1#\1#'); [ "$R" = 1 ] && [ "${T%.*}" = 4 ] && break; sleep 5; done
echo "$(date -u +%FT%TZ) sampler pods: $R, ray tpu-group-1 total: $T"; $K get pods -l ray.io/cluster=$C -o wide --no-headers | awk '{print "  ", $1, $3, $7}'
nohup ./sandbox_monitor.sh "$JOB" 20 > /dev/null 2>&1 &
python3 mkenv.py "$JOB" "{\"min\": 1, \"max\": $MX, \"max_rank\": 8, \"initial_active\": 1, \"log\": \"/data/tpu-m3/pool-$JOB.jsonl\", \"port\": 8787, \"lb_poll_s\": 2.0, \"health_poll_s\": 3.0, \"health_timeout_s\": 5.0, \"health_failures\": 2, \"presync_timeout_s\": 5.0, \"probe\": false, \"spawn\": true, \"early_spawn\": true, \"join_on_ready\": true, \"release_on_scale_in\": true}" >/dev/null
CLUSTER=$C ./submit.sh "$JOB" "runtime_env.$JOB.json" ROLLOUT_NODES=1 ROLLOUT_CHIPS=4 ROLLOUT_TP=4 TRAIN_TP=1 TRAIN_DP=4 TRAIN_BS="$BS" ROLLOUT_N=4 STEPS=60 MAX_NUM_SEQS=128 EXP_NAME="$JOB" 2>&1 | tail -2
echo "$(date -u +%FT%TZ) dynamic job $JOB submitted on $C (head + trainer + ONE sampler, nothing pre-created)"
sleep 45
CLUSTER=$C ./run_controller.sh "$JOB" MAX_STANDALONE=$MX IDLE_HI=0.65 IDLE_LO=0.30 UTIL_HI=0.75 UTIL_LO=0.40 FLOOR_MIN_GAIN=0.05 FLOOR_SKIP_VERSIONS=2 DWELL_STEPS=2 MIN_ACTION_INTERVAL_S=120 POLL_INTERVAL_S=10
