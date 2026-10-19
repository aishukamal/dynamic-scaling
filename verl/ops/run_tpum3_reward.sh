#!/usr/bin/env bash
# run_tpum3_reward.sh : reward-work wrapper around the dyn3 recipe. Same script, same pins; the only differences are the env knobs
# (TRAIN_BS, ROLLOUT_N, STEPS, EXP_NAME, SWE_* loop knobs) and the hydra overrides in EXTRA_HYDRA (space separated, no spaces inside).
set -euo pipefail
echo "reward wrapper: EXTRA_HYDRA=${EXTRA_HYDRA:-} SWE_BLOCK_POLICY=${SWE_BLOCK_POLICY:-unset} SWE_SUBMIT_GUARD=${SWE_SUBMIT_GUARD:-unset} SWE_TURN_DUMP=${SWE_TURN_DUMP:-unset}"
cat /overlay/swe_work/SWE_LOOP_SOURCE 2>/dev/null || true
# OPT_OFFLOAD=True keeps the Adam states in host memory (memory placement only, numerics unchanged): the 4B FSDP=4 trainer hit an
# HBM fragmentation OOM in update_actor at step 2 with 29.7 GB of 31.2 GB per chip in use (tpum3-4bb). Same for PARAM_OFFLOAD.
# LR (default the recipe's 1e-6) and SAVE_FREQ (default -1 = no checkpoints) for the learning experiment (tpum3-4be: lr 3e-6, save at 30).
sed -e "s/torchtitan.optimizer_offload=False/torchtitan.optimizer_offload=${OPT_OFFLOAD:-False}/" \
    -e "s/torchtitan.param_offload=False/torchtitan.param_offload=${PARAM_OFFLOAD:-False}/" \
    -e "s/actor.optim.lr=1e-6/actor.optim.lr=${LR:-1e-6}/" \
    -e "s/trainer.save_freq=-1/trainer.save_freq=${SAVE_FREQ:--1}/" \
    -e "s/ppo_max_token_len_per_gpu=\"\${MAX_LEN}\"/ppo_max_token_len_per_gpu=\"\${PPO_MAX_TOKENS:-\${MAX_LEN}}\"/" \
    -e "s/actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1/actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGP_MBS:-1}/" \
    -e "s/actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1/actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGP_MBS:-1}/" \
    /data/tpu-m3/run_tpum3_swe_disagg.sh > /tmp/run_tpum3_swe_disagg.reward.sh
grep -o "optimizer_offload=[A-Za-z]*\|param_offload=[A-Za-z]*\|optim.lr=[0-9e.-]*\|save_freq=[0-9-]*\|micro_batch_size_per_gpu=[0-9]*\|use_dynamic_bsz=[A-Za-z]*" /tmp/run_tpum3_swe_disagg.reward.sh | sort -u | tr '\n' ' '; echo
# shellcheck disable=SC2086
exec bash /tmp/run_tpum3_swe_disagg.reward.sh ${EXTRA_HYDRA:-}
