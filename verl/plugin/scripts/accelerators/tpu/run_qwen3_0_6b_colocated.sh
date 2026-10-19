#!/usr/bin/env bash
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# GRPO | Qwen3-0.6B | GSM8K | trainer and rollout COLOCATED on one TPU host (v5e-8 or v6e-8, 8 chips)
#
# verl's default sync V1 trainer (trainer_mode sync; colocate_async also works) with hybrid_engine=True and the
# plugin's colocated checkpoint engine manager: the TorchTitan trainer (FSDP over all 8 chips) and one vLLM replica of TP=8 alternate on the same
# chips through libtpu's in-process suspend/resume. Bring-up smoke: 3 steps, rewards and the per-sync
# timings in the job log. Needs LIBTPU_CHECKPOINTING_ENABLED=true on every TPU process.
#
# ROLLOUT_TP defaults to the whole host: a hybrid replica gets the trainer bundle's chips in rank order, and
# on a v5e-8 host the chip ids are interleaved over the physical 2x4, so chips 0-3 and 4-7 form neither a 2x2
# block nor a 1x4 row (libtpu: "Failed to find any 2x2 corner", "Mesh build was incomplete"). Sub-host
# replicas need a chip set that is a physical mesh; the multi-host recipes use whole hosts (v6e-32: TP=4 = one
# 4-chip host) or host pairs (v5e-64: TP=8).
set -xeuo pipefail

export VERL_PLATFORM=tpu RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1 RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export LIBTPU_CHECKPOINTING_ENABLED=true
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_use_enhanced_launch_barrier=false}"
# Dynamo recompile limit for the compiled TransformerBlocks: the bucketed micro-batch shapes exceed Dynamo's own 8 static shapes.
export VERL_TPU_DYNAMO_RECOMPILE_LIMIT="${VERL_TPU_DYNAMO_RECOMPILE_LIMIT:-64}"

N_CHIPS="${N_CHIPS:-8}"; NNODES="${NNODES:-1}"; ROLLOUT_TP="${ROLLOUT_TP:-${N_CHIPS}}"
MODEL_PATH="${MODEL_PATH:-/data/assets/hf/Qwen3-0.6B}"
TRAIN_FILE="${TRAIN_FILE:-/data/data/gsm8k/train.parquet}"; TEST_FILE="${TEST_FILE:-/data/data/gsm8k/test.parquet}"
STEPS="${STEPS:-3}"
MANAGER=verl_hardware_plugin.accelerators.tpu.engines.tpu_colocated.TPUColocatedCheckpointEngineManager

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    transfer_queue.enable=True model_engine=torchtitan \
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" data.val_files="${TEST_FILE}" \
    data.train_batch_size=8 data.val_batch_size=8 data.val_max_samples=16 \
    data.max_prompt_length=512 data.max_response_length=512 \
    +data.max_length=4096 +data.max_token_len_per_gpu=4096 \
    data.filter_overlong_prompts=True data.truncation=error +data.pad_mode=no_padding \
    actor_rollout_ref.actor.strategy=torchtitan actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.actor.torchtitan.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.torchtitan.param_offload=False actor_rollout_ref.actor.torchtitan.optimizer_offload=False \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size=1 actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${N_CHIPS}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    +actor_rollout_ref.ref_in_actor=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.load_format=safetensors actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=raiden \
    ++actor_rollout_ref.rollout.checkpoint_manager_class="${MANAGER}" \
    trainer.val_before_train=False trainer.logger="['console']" \
    trainer.project_name=verl_tpu_grpo trainer.experiment_name=qwen3_0_6b_colocated_smoke \
    trainer.save_freq=-1 trainer.test_freq=-1 trainer.total_epochs=1 trainer.total_training_steps="${STEPS}" \
    trainer.nnodes="${NNODES}" trainer.n_gpus_per_node="${N_CHIPS}" "$@"
