#!/usr/bin/env bash
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# GRPO | Qwen3-14B | GSM8K | trainer and rollout COLOCATED on one slice (the locked GPU-vs-TPU recipe)
#
# SLICE=v6e-32 (default): ct6e-standard-4t 4x8 = 8 hosts x 4 chips; FSDP dp=32; 8 vLLM replicas of TP=4 (one per
#   host); 4096 trainer tokens per chip; KV budget 10,098 MiB per rank (set GPU_MEM_UTIL so vLLM's memory line
#   reports it).
# SLICE=v5e-64: ct5lp-hightpu-4t 8x8 = 16 hosts x 4 chips; FSDP dp=64; 8 replicas of TP=8 (one per machine pair);
#   2048 trainer tokens per chip; KV budget 4,117.5 MiB per rank.
# Common: 256 prompts x n=8, 512/512, ppo_mini_batch_size 256 (one optimizer step per step), lr 1e-6, GRPO with
# low_var_kl 0.001, prefix caching off, max_model_len 1024, max_num_seqs 256, ref model on, 11 steps, validation
# at step 10 on the full GSM8K test set. Runs on verl's default sync V1 trainer (trainer_mode sync; colocate_async also works)
# with hybrid_engine=True and the plugin's colocated checkpoint engine manager. Needs LIBTPU_CHECKPOINTING_ENABLED=true on
# every TPU process.
set -xeuo pipefail

export VERL_PLATFORM=tpu RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1 RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export LIBTPU_CHECKPOINTING_ENABLED=true
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_scoped_vmem_limit_kib=65536}"
export XLA_FLAGS="${XLA_FLAGS:---xla_disable_hlo_passes=instruction-fusion,fusion-merger,multi-output-fusion,horizontal-fusion}"
export VERL_TPU_SEQ_BUCKET_SIZE="${VERL_TPU_SEQ_BUCKET_SIZE:-1024}"
# Dynamo recompile limit for the compiled TransformerBlocks: the bucketed micro-batch shapes exceed Dynamo's own 8 static shapes.
export VERL_TPU_DYNAMO_RECOMPILE_LIMIT="${VERL_TPU_DYNAMO_RECOMPILE_LIMIT:-64}"

SLICE="${SLICE:-v6e-32}"
case "${SLICE}" in
  v6e-32) NNODES=8;  N_CHIPS=4; ROLLOUT_TP="${ROLLOUT_TP:-4}"; TOK="${TOK:-4096}"; GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.545}" ;;
  v5e-64) NNODES=16; N_CHIPS=4; ROLLOUT_TP="${ROLLOUT_TP:-8}"; TOK="${TOK:-2048}"; GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.49}" ;;
  *) echo "SLICE must be v6e-32 or v5e-64" >&2; exit 1 ;;
esac
TRAIN_CHIPS=$((NNODES * N_CHIPS))
MODEL_PATH="${MODEL_PATH:-/data/assets/hf/Qwen3-14B}"
TRAIN_FILE="${TRAIN_FILE:-/data/data/gsm8k/train.parquet}"; TEST_FILE="${TEST_FILE:-/data/data/gsm8k/test.parquet}"
STEPS="${STEPS:-11}"; TEST_FREQ="${TEST_FREQ:-10}"; MICRO="${MICRO:-4}"
MANAGER=verl_hardware_plugin.accelerators.tpu.engines.tpu_colocated.TPUColocatedCheckpointEngineManager
EXP_NAME="${EXP_NAME:-qwen3_14b_colocated_${SLICE}}"

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    transfer_queue.enable=True model_engine=torchtitan \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" data.val_files="${TEST_FILE}" \
    data.train_batch_size=256 data.val_batch_size=64 \
    data.max_prompt_length=512 data.max_response_length=512 \
    +data.max_length=1024 +data.max_token_len_per_gpu="${TOK}" \
    data.filter_overlong_prompts=True data.truncation=error +data.pad_mode=no_padding \
    actor_rollout_ref.actor.strategy=torchtitan actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=True actor_rollout_ref.actor.torchtitan.use_torch_compile=True \
    ++actor_rollout_ref.actor.torchtitan.use_splash_attention=True ++actor_rollout_ref.actor.torchtitan.use_simple_fsdp=True \
    ++actor_rollout_ref.actor.torchtitan.tpu_eager_mode=DEFER_AND_FUSE \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint=full \
    actor_rollout_ref.actor.torchtitan.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${TOK}" \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 ++actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.torchtitan.param_offload=False actor_rollout_ref.actor.torchtitan.optimizer_offload=False \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size=1 actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${TRAIN_CHIPS}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO}" actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${TOK}" \
    actor_rollout_ref.ref.torchtitan.use_torch_compile=True ++actor_rollout_ref.ref.torchtitan.use_splash_attention=True \
    ++actor_rollout_ref.ref.torchtitan.use_simple_fsdp=True ++actor_rollout_ref.ref.torchtitan.tpu_eager_mode=DEFER_AND_FUSE \
    actor_rollout_ref.ref.torchtitan.tensor_parallel_size=1 actor_rollout_ref.ref.torchtitan.data_parallel_shard_size="${TRAIN_CHIPS}" \
    actor_rollout_ref.ref.torchtitan.pipeline_parallel_size=1 actor_rollout_ref.ref.torchtitan.attn_type=varlen \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_prefix_caching=False +actor_rollout_ref.rollout.engine_kwargs.vllm.no_enable_prefix_caching=True \
    actor_rollout_ref.rollout.load_format=safetensors actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len=1024 actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO}" actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${TOK}" \
    actor_rollout_ref.rollout.checkpoint_engine.backend=raiden \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.verify_parity=False \
    ++actor_rollout_ref.rollout.checkpoint_manager_class="${MANAGER}" \
    trainer.val_before_train=False trainer.logger="['console','tensorboard']" \
    trainer.project_name=verl_tpu_grpo trainer.experiment_name="${EXP_NAME}" \
    trainer.save_freq=-1 trainer.test_freq="${TEST_FREQ}" trainer.total_epochs=10 trainer.total_training_steps="${STEPS}" \
    trainer.nnodes="${NNODES}" trainer.n_gpus_per_node="${N_CHIPS}" "$@"
