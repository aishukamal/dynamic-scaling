#!/usr/bin/env bash
# go_static_r2.sh <cluster> <job> <N samplers> [versions=8] : recipe-2 static reference arm (8:4N) on an EXISTING cluster with the
# document-series settings frozen at dynr3 (plugin 3a4b8e1, loop v3.2, 512/version, 2048/6144, hook timeouts 20 s, hung rule off,
# SWE_API_CONCURRENCY 32). Patches tpu-group-1 to N replicas, waits for N sampler pods + Ray TPUs, submits with ROLLOUT_NODES=N
# (all N active from the start, spawn off), starts the sandbox monitor and a stop_after watcher for <versions> versions. No controller.
set -euo pipefail
C=$1; JOB=$2; N=$3; V=${4:-8}
D="$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3"; cd "$D"
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=120s"
IDX=$($K get raycluster $C -o json | python3 -c "import sys,json; g=json.load(sys.stdin)['spec']['workerGroupSpecs']; print([x['groupName'] for x in g].index('tpu-group-1'))")
$K patch raycluster $C --type=json -p "[{\"op\":\"replace\",\"path\":\"/spec/workerGroupSpecs/$IDX/replicas\",\"value\":$N},{\"op\":\"replace\",\"path\":\"/spec/workerGroupSpecs/$IDX/minReplicas\",\"value\":1},{\"op\":\"remove\",\"path\":\"/spec/workerGroupSpecs/$IDX/scaleStrategy\"}]" 2>/dev/null || \
$K patch raycluster $C --type=json -p "[{\"op\":\"replace\",\"path\":\"/spec/workerGroupSpecs/$IDX/replicas\",\"value\":$N},{\"op\":\"replace\",\"path\":\"/spec/workerGroupSpecs/$IDX/minReplicas\",\"value\":1}]"
for i in $(seq 1 80); do R=$($K get pods -l ray.io/cluster=$C,ray.io/group=tpu-group-1 --no-headers 2>/dev/null | awk '$3=="Running"' | wc -l | tr -d ' '); [ "$R" -ge "$N" ] && break; sleep 5; done
echo "$(date -u +%FT%TZ) $C sampler pods Running: $R"; $K get pods -l ray.io/cluster=$C -o wide --no-headers | awk '{print "  ", $1, $3, $7}'
HEAD=$($K get pods -l ray.io/node-type=head,ray.io/cluster=$C -o jsonpath='{.items[0].metadata.name}')
HEADHOST=$($K get pods -l ray.io/node-type=head,ray.io/cluster=$C -o jsonpath='{.items[0].spec.nodeName}' | sed -E 's/.*-//')
for i in $(seq 1 60); do T=$($K exec "$HEAD" -- ray status 2>/dev/null | grep -oE "^ [0-9.]+/[0-9.]+ TPU" | sed -E 's#.*/([0-9.]+) TPU#\1#'); [ "${T%.*}" -ge $((8 + N*4)) ] 2>/dev/null && break; sleep 5; done
echo "$(date -u +%FT%TZ) ray TPU total: $T (need $((8 + N*4)))"
echo "pre-launch: sandboxes=$($K get sandboxes -A --no-headers 2>/dev/null | wc -l | tr -d ' ')"
RT_BASE=runtime_env.r2base.json python3 mkenv.py "$JOB" "{\"min\": 1, \"max\": $N, \"max_rank\": 8, \"initial_active\": $N, \"log\": \"/data/tpu-m3/pool-$JOB.jsonl\", \"port\": 8787, \"lb_poll_s\": 2.0, \"health_poll_s\": 3.0, \"health_timeout_s\": 20.0, \"health_failures\": 2, \"hung_after_s\": 86400, \"presync_timeout_s\": 20.0, \"probe\": false, \"spawn\": false, \"early_spawn\": false, \"join_on_ready\": true}" SWE_TURN_DUMP=/data/tpu-m3/turns-$JOB SWE_SUBMIT_GUARD=1 SWE_REPEAT_GUARD=1 SWE_BLOCK_POLICY=first SWE_API_CONCURRENCY=32 PPO_MAX_TOKENS=8192 EXTRA_HYDRA="+data.apply_chat_template_kwargs.enable_thinking=false actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True" >/dev/null
nohup ./sandbox_monitor.sh "$JOB" 20 > /dev/null 2>&1 &
CLUSTER=$C SCRIPT=run_tpum3_reward.sh ./submit.sh "$JOB" "runtime_env.$JOB.json" MODEL_PATH=/tmp/tunix-models/Qwen3-4B PROMPT_LEN=2048 RESPONSE_LEN=6144 TRAIN_FILE=/data/data/swe/train_calibrated_eu.parquet TEST_FILE=/data/data/swe/test_eu.parquet ROLLOUT_NODES=$N ROLLOUT_CHIPS=4 ROLLOUT_TP=4 TRAIN_CHIPS=8 TRAIN_TP=1 TRAIN_DP=8 TRAIN_BS=128 ROLLOUT_N=4 MICRO_BS=1 STEPS=60 MAX_NUM_SEQS=128 EXP_NAME="$JOB" 2>&1 | tail -2
(CLUSTER=$C nohup ./stop_after.sh "$JOB" "tpum3-stage-$HEADHOST" "/data/tpu-m3/pool-$JOB.jsonl" "$V" > "logs/stop_after-$JOB.out" 2>&1 &)
echo "$(date -u +%FT%TZ) static arm $JOB (S=$N) on $C started (stop after $V versions)"
