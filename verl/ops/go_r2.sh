#!/usr/bin/env bash
# go_r2.sh <cluster> <job> <TRAIN_BS> <PROMPT_LEN> <RESPONSE_LEN> [IDLE_HI UTIL_HI UTIL_LO] : recipe-2 dynamic arm = cold product path from S=1 with the
# pool hook (early spawn, on-demand joiner push, zero-replica sync skip, single-spawn fix, 20 s timeouts, hung rule off, API sem 32), the controller with the K8s actuator, the 24-version
# stopper and the version-13 kill. The sampler group of <cluster> must have replicas 1 (scale it before calling). Band defaults = the re-derived
# recipe-2 band (IDLE 0.05/0.0, UTIL 0.70/0.40); pass IDLE_HI UTIL_HI to override (IDLE_LO/UTIL_LO follow the dyn3 ratios).
set -euo pipefail
C=$1; JOB=$2; BS=$3; PL=$4; RL=$5; IH=${6:-0.05}; UH=${7:-0.70}; UL=${8:-0.40}
D="$HOME/workspaces/rl-pool-autoscaler-ops/tpu-m3"; cd "$D"
CTX=gke_aishuk-test_europe-west4-b_tpu-stress-test-eu; K="/opt/homebrew/bin/kubectl --context $CTX --request-timeout=120s"
HEADHOST=$($K get pods -l ray.io/node-type=head,ray.io/cluster=$C -o jsonpath='{.items[0].spec.nodeName}' | sed -E 's/.*-//')
RT_BASE=runtime_env.r2base.json python3 mkenv.py "$JOB" "{\"min\": 1, \"max\": 3, \"max_rank\": 8, \"initial_active\": 1, \"log\": \"/data/tpu-m3/pool-$JOB.jsonl\", \"port\": 8787, \"lb_poll_s\": 2.0, \"health_poll_s\": 3.0, \"health_timeout_s\": 20.0, \"health_failures\": 2, \"hung_after_s\": 86400, \"presync_timeout_s\": 20.0, \"probe\": false, \"spawn\": true, \"early_spawn\": true, \"join_on_ready\": true, \"release_on_scale_in\": true}" SWE_TURN_DUMP=/data/tpu-m3/turns-$JOB SWE_SUBMIT_GUARD=1 SWE_REPEAT_GUARD=1 SWE_BLOCK_POLICY=first SWE_API_CONCURRENCY=32 PPO_MAX_TOKENS=8192 EXTRA_HYDRA="+data.apply_chat_template_kwargs.enable_thinking=false actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True" >/dev/null
nohup ./sandbox_monitor.sh "$JOB" 20 > /dev/null 2>&1 &
CLUSTER=$C SCRIPT=run_tpum3_reward.sh ./submit.sh "$JOB" "runtime_env.$JOB.json" MODEL_PATH=/tmp/tunix-models/Qwen3-4B PROMPT_LEN="$PL" RESPONSE_LEN="$RL" TRAIN_FILE=/data/data/swe/train_calibrated_eu.parquet TEST_FILE=/data/data/swe/test_eu.parquet ROLLOUT_NODES=1 ROLLOUT_CHIPS=4 ROLLOUT_TP=4 TRAIN_CHIPS=8 TRAIN_TP=1 TRAIN_DP=8 TRAIN_BS="$BS" ROLLOUT_N=4 MICRO_BS=1 STEPS=60 MAX_NUM_SEQS=128 EXP_NAME="$JOB" 2>&1 | tail -2
echo "$(date -u +%FT%TZ) dynamic arm $JOB submitted on $C (head + trainer + ONE sampler)"
sleep 45
CLUSTER=$C ./run_controller.sh "$JOB" MAX_STANDALONE=3 IDLE_HI=$IH IDLE_LO=0.0 UTIL_HI=$UH UTIL_LO=$UL FLOOR_MIN_GAIN=0.05 FLOOR_SKIP_VERSIONS=2 DWELL_STEPS=2 MIN_ACTION_INTERVAL_S=120 POLL_INTERVAL_S=10 2>&1 | tail -1 | cut -c1-120
(CLUSTER=$C nohup ./stop_after.sh "$JOB" "tpum3-stage-$HEADHOST" "/data/tpu-m3/pool-$JOB.jsonl" 24 > "logs/stop_after-$JOB.out" 2>&1 &)
(JOB=$JOB CLUSTER=$C nohup ./kill_at_v13_job.sh > "logs/kill_at_v13-$JOB.out" 2>&1 &)
echo "$(date -u +%FT%TZ) controller, stopper (24) and kill@v13 armed for $JOB"
