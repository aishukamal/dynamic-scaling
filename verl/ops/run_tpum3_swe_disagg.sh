#!/usr/bin/env bash
# TPU-M3: verl V1 separate_async on torchtpu, DISAGGREGATED, SWE agentic workload (py-inference-scheduler SWEAgentLoop + gVisor
# agent-sandboxes). Trainer: TorchTitan FSDP over TRAIN_CHIPS chips (one v6e-4 host). Samplers: ROLLOUT_NODES hosts x ROLLOUT_CHIPS
# chips, TP=ROLLOUT_TP standalone vLLM-TPU replicas (one replica per host at TP=4), Raiden weight sync, no C/R.
# Skeleton = TPU-M2 run_tpum2_disagg.sh (V1 knobs, torchtitan, raiden); algorithm + generation knobs = GPU M0 recipe
# run_m0_swe_disagg.sh (grpo, lr 1e-6, clip_ratio_high 0.28, no KL, entropy 0, prompt 4096 / response 16384, n 4, max 32 turns,
# multi_turn.enable, swe_agent_loop.yaml). Trajectories per step = TRAIN_BS x ROLLOUT_N (one update per step: mini = TRAIN_BS).
set -xeuo pipefail
export VERL_PLATFORM=tpu RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1 RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_use_enhanced_launch_barrier=false}"
export VERL_TPU_DYNAMO_RECOMPILE_LIMIT="${VERL_TPU_DYNAMO_RECOMPILE_LIMIT:-64}"
TRAIN_CHIPS="${TRAIN_CHIPS:-4}"; TRAIN_TP="${TRAIN_TP:-1}"; TRAIN_DP="${TRAIN_DP:-4}"; ROLLOUT_NODES="${ROLLOUT_NODES:-1}"; ROLLOUT_CHIPS="${ROLLOUT_CHIPS:-4}"; ROLLOUT_TP="${ROLLOUT_TP:-4}"
STEPS="${STEPS:-8}"; TRAIN_BS="${TRAIN_BS:-32}"; ROLLOUT_N="${ROLLOUT_N:-4}"; MICRO_BS="${MICRO_BS:-1}"
PROMPT_LEN="${PROMPT_LEN:-4096}"; RESPONSE_LEN="${RESPONSE_LEN:-8192}"; MAX_TURNS="${MAX_TURNS:-32}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"; MAX_BATCHED="${MAX_BATCHED:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"; VERIFY_PARITY="${VERIFY_PARITY:-False}"; EXP_NAME="${EXP_NAME:-tpum3}"
MODEL_PATH="${MODEL_PATH:-/tmp/tunix-models/Qwen3-1.7B}"
TRAIN_FILE="${TRAIN_FILE:-/data/data/swe/train_calibrated.parquet}"; TEST_FILE="${TEST_FILE:-/data/data/swe/test.parquet}"
MAX_LEN=$(( PROMPT_LEN + RESPONSE_LEN ))
export SWE_SANDBOX_NAMESPACE="${SWE_SANDBOX_NAMESPACE:-agents-system}"
export SWE_CMD_TIMEOUT_S="${SWE_CMD_TIMEOUT_S:-60}" SWE_OBS_MAX_CHARS="${SWE_OBS_MAX_CHARS:-6000}"
export SWE_SANDBOX_CPU_REQUEST="${SWE_SANDBOX_CPU_REQUEST:-200m}" SWE_SANDBOX_MEM_REQUEST="${SWE_SANDBOX_MEM_REQUEST:-768Mi}"
export SWE_SANDBOX_WAIT_S="${SWE_SANDBOX_WAIT_S:-600}"
echo "TPUM3 layout: TRAIN_CHIPS=$TRAIN_CHIPS TRAIN_TP=$TRAIN_TP TRAIN_DP=$TRAIN_DP ROLLOUT_NODES=$ROLLOUT_NODES ROLLOUT_CHIPS=$ROLLOUT_CHIPS ROLLOUT_TP=$ROLLOUT_TP TRAIN_BS=$TRAIN_BS N=$ROLLOUT_N MAX_NUM_SEQS=$MAX_NUM_SEQS POOL=${VERL_TPU_POOL_CONTROL:-unset} PYTHONPATH=${PYTHONPATH:-unset}"
cat /verl-hardware-plugin/PLUGIN_COMMIT /verl/VERL_COMMIT /opt/swe_work/SWE_LOOP_SOURCE 2>/dev/null || true

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=1 \
    transfer_queue.enable=True model_engine=torchtitan \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" data.val_files="${TEST_FILE}" \
    data.return_raw_chat=True \
    data.train_batch_size="${TRAIN_BS}" data.val_batch_size="${TRAIN_BS}" data.val_max_samples=16 \
    data.max_prompt_length="${PROMPT_LEN}" data.max_response_length="${RESPONSE_LEN}" \
    +data.max_length="${MAX_LEN}" +data.max_token_len_per_gpu="${MAX_LEN}" \
    data.filter_overlong_prompts=True data.truncation=error +data.pad_mode=no_padding \
    actor_rollout_ref.actor.strategy=torchtitan actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.actor.torchtitan.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BS}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BS}" actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_LEN}" \
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.torchtitan.param_offload=False actor_rollout_ref.actor.torchtitan.optimizer_offload=False \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TRAIN_TP}" actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${TRAIN_DP}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.actor.torchtitan.max_seq_len="${MAX_LEN}" actor_rollout_ref.ref.torchtitan.max_seq_len="${MAX_LEN}" \
    actor_rollout_ref.actor.torchtitan.entropy_from_logits_with_chunking=True actor_rollout_ref.ref.torchtitan.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BS}" actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_LEN}" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.load_format=safetensors actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.max_model_len="${MAX_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_BATCHED}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BS}" actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_LEN}" \
    actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
    +actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_YAML:-/opt/swe_work/integration/verl/examples/swe_agent_loop.yaml}" \
    actor_rollout_ref.rollout.checkpoint_engine.backend=raiden \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.verify_parity="${VERIFY_PARITY}" \
    actor_rollout_ref.rollout.nnodes="${ROLLOUT_NODES}" actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_CHIPS}" \
    trainer.val_before_train=False trainer.logger="['console']" \
    trainer.project_name=rl_pool_autoscaler trainer.experiment_name="${EXP_NAME}" \
    trainer.save_freq=-1 trainer.test_freq=-1 trainer.total_epochs=1000 trainer.total_training_steps="${STEPS}" \
    trainer.nnodes=1 trainer.n_gpus_per_node="${TRAIN_CHIPS}" "$@"
