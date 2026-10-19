# Google TPU Colocated Mode

Trainer and rollout on one TPU slice, driven by verl's default sync V1 trainer with hybrid vLLM replicas on the
trainer's own workers (`trainer.use_v1=True`, `hybrid_engine=True`; `trainer.v1.trainer_mode=colocate_async` also
works), the loop the GPU hybrid engine uses. Nothing in the loop changes; the TPU specifics live in a checkpoint engine manager subclass and in the vLLM rollout
glue of this plugin.

## Why a subclass

On GPU one CUDA device can be open to the trainer and to vLLM at the same time, so the stock manager only
sleeps and wakes vLLM around the training phases. A TPU chip has one owner process. The colocated manager
(`verl_hardware_plugin.accelerators.tpu.engines.tpu_colocated.TPUColocatedCheckpointEngineManager`)
therefore composes three concerns, each with one meaning:

| Concern | GPU today | TPU in this plugin |
|---------|-----------|--------------------|
| Memory | vLLM `sleep(level)` / `wake_up(tags)` | The same calls; `rollout/tpu_sleep.py` gives vllm-torchtpu's `TPUWorker` a memory-only sleep mode until it ships one. Level 1 copies the weights to host and frees the KV cache, level 2 frees both; `wake_up` re-allocates. The server picks level 2 for the `raiden` and `tpu` weight syncs. |
| tpu-sync receive buffers | n/a | `drop_receive_buffers` / `rebind_receive_buffers` on the Raiden worker extension, through `collective_rpc`. |
| The device | never released | `platform_tpu.suspend_device` / `resume_device` (libtpu's in-process checkpoint / restore: HBM to host memory, device descriptors closed, process alive, compiled programs kept). On the rollout through the extension's `suspend_device` / `resume_device`; on the trainer through `worker.__ray_call__` on every trainer worker actor. |

Sequence (`T` = trainer workers, `S` = rollout workers):

- `sleep_replicas` (after sampling): S vLLM sleep(2) -> S drop receive buffers -> S release chips -> T re-attach chips
- `update_weights` (end of step): T stage + transfer into S's host buffers -> T release send buffers -> T release chips ->
  S re-attach chips -> S `wake_up(["weights"])` -> S rebind receive buffers -> S install -> S `wake_up(["kv_cache"])` ->
  tag the weight version -> resume generation

The first sync adds one round trip so the rollout workers can register their receive buffers while they hold the
chips. At any instant at most one of the two process families holds the chips.

Placement follows upstream's hybrid server: vLLM's workers are scheduled by node affinity to the trainer workers'
nodes with no resource claim and the trainer worker's chip in `TPU_VISIBLE_CHIPS`, as the GPU server inherits
`CUDA_VISIBLE_DEVICES`. No placement group and no resource pool are involved. A replica that spans several hosts
(`tensor_model_parallel_size` larger than the chips of one host, e.g. TP=8 on 4-chip v5e hosts) is formed from
hosts of one physical machine: the hosts are ordered by `VERL_TPU_HOST_ORDER` (comma-separated host IPs in
physical order) when set, else grouped by the VM metadata attribute `physical_host` (the two VMs of one machine
report the same value), and consecutive hosts in that order form a replica; rank order alone pairs hosts by IP,
which on a multi-host slice is not a physical pair and libtpu's SliceBuilder then fails to build the mesh. Several replicas per host are the
rule (for example 8 replicas of TP=4 on a 32-chip slice, one per 4-chip host); all of their workers register under the one
Raiden job `sampler` with global unit ids `replica_idx * TP + tp_rank`, and every sync is one controller transfer to all of them.

## Requirements

- `LIBTPU_CHECKPOINTING_ENABLED=true` in the environment of every TPU process (trainer workers and vLLM workers):
  it starts the libtpu control thread the device release talks to. Verify a host with
  `scripts/accelerators/tpu/selfcr_smoke.py --runtime torch` (suspend, resume, bit-exact recompute).
- The pins of the [vLLM rollout](./rollout.md) and the [weight sync](./weight_sync.md), with a `tpu-sync-torch`
  that has `WeightSynchronizer.unbind_weights()` (`0.0.1.dev20261006025601` or newer, built for the installed
  `torch-tpu`). Without it the receive buffers stay in the suspended image (logged once) and the trainer falls
  back to re-creating its synchronizer every sync.
- `actor_rollout_ref.actor.torchtitan.param_offload=False` and `optimizer_offload=False` (the defaults): the
  engine's tensor offload is not used; the device release is the only hand-over.
- `actor_rollout_ref.rollout.free_cache_engine=True` (the default): upstream gates the hybrid sleep and the KV
  wake on it.

## Configuration

```bash
trainer.use_v1=True                                        # trainer_mode defaults to sync; colocate_async also works
actor_rollout_ref.hybrid_engine=True
actor_rollout_ref.rollout.checkpoint_engine.backend=raiden
++actor_rollout_ref.rollout.checkpoint_manager_class=verl_hardware_plugin.accelerators.tpu.engines.tpu_colocated.TPUColocatedCheckpointEngineManager
actor_rollout_ref.rollout.tensor_model_parallel_size=4     # chips per replica; replicas = trainer chips / TP
```

`checkpoint_manager_class` is declared in verl's `RolloutConfig` dataclass but not in `rollout.yaml`, so it is passed with hydra's `++` (add or override); the plugin's TorchTitan engine fields (`use_splash_attention`, `use_simple_fsdp`, `tpu_eager_mode`) are added to the dataclass by the plugin and take `++` for the same reason. `checkpoint_manager_class` is honoured by a verl whose V1 trainer resolves that key. On an older verl the plugin
binds a selector into the V1 trainer module when the first hybrid replica is created
(`tpu_colocated.install_manager_selector`), which picks the colocated manager for TPU hybrid replicas with the
`raiden` or `tpu` backend and the stock manager otherwise; the configured backend is re-read from the rollout
config because the trainer forces `naive` on the stock class.

Dynamic batch sizing (`use_dynamic_bsz=True`) agrees micro-batch counts across the data-parallel ranks with
MAX / MIN all-reduces of small integers; torch_tpu's `tpu_dist` lowers only SUM, so `PlatformTPU.all_reduce`
(verl's `PlatformBase.all_reduce` hook) carries them through one SUM of a per-rank row matrix. Without the hook in
verl the step fails at the first log-prob pass with `supported lowering only of Sum all reduce`.

Environment: `VERL_TPU_TOPOLOGY_<n>` (per-size topology override, unset by default) and
`VERL_TPU_DYNAMO_RECOMPILE_LIMIT` (Dynamo recompile limit for the compiled TransformerBlocks; default 64, the value
the plugin applies anyway; the recipes export 64 explicitly because the bucketed micro-batch shapes exceed Dynamo's
own limit of 8 static shapes).

## Recipes

Both scripts are under `scripts/accelerators/tpu/` and submit through the same Ray job command as the
[vLLM rollout](./rollout.md) example, with `LIBTPU_CHECKPOINTING_ENABLED=true` (and the
`VERL_TPU_DYNAMO_RECOMPILE_LIMIT=64` the scripts export) added to the job's `env_vars` and to the worker pod environment:

- `run_qwen3_0_6b_colocated.sh`: the bring-up smoke on one v5e-8 or v6e-8 host (3 steps, one replica of TP=8 by
  default, rewards and the per-sync timings in the log). A hybrid replica takes the trainer bundle's chips in rank
  order; on an 8-chip v5e host the chip ids are interleaved over the physical 2x4, so the 4-chip halves do not form a
  mesh (libtpu `Failed to find any 2x2 corner` / `Mesh build was incomplete`): sub-host replicas need a chip set
  that is a physical mesh, which this path does not choose. Several replicas per host work when each replica is a
  whole host or a host pair (the 14B recipes).
- `run_qwen3_14b_colocated.sh`: Qwen3-14B GRPO on GSM8K with the locked comparison recipe (256 prompts x n=8,
  512/512, one optimizer step per step, lr 1e-6, prefix caching off, 11 steps, validation at step 10).
  `SLICE=v6e-32` (default: 8 hosts x 4 chips, 8 replicas of TP=4, 4096 trainer tokens per chip) or
  `SLICE=v5e-64` (16 hosts x 4 chips, 8 replicas of TP=8, 2048 trainer tokens per chip). Set
  `GPU_MEM_UTIL` so that vLLM's memory line reports the locked KV budget (10,098 MiB per rank on v6e,
  4,117.5 MiB per rank on v5e-64).

## Metrics

`update_weights` returns the Raiden phases (`timing_s/tpu-sync/trainer_init`, `barrier`, `p2p_transfer`,
`trainer_release`, `sampler_h2d`, `sampler_h2d_pure`, `sampler_init` on the first sync, `total_sync`) plus the
hand-over phases: `timing_s/tpu-colocated/sampler_sleep`, `sampler_drop`, `sampler_suspend`, `trainer_resume`
(recorded in `sleep_replicas`, reported with the next sync), `trainer_suspend`, `sampler_resume`, `wake_weights`,
`sampler_rebind`, `wake_kv`. The driver log has one line per sync with the same numbers and one line per
hand-over with the MiB the rollout workers dropped and re-allocated.

## Verify

On any host, without vLLM, tpu-sync or a TPU:

```bash
pytest tests/accelerators/tpu/test_tpu_control.py tests/accelerators/tpu/test_tpu_sleep.py \
       tests/accelerators/tpu/test_tpu_colocated.py tests/accelerators/tpu/test_tpu_vllm_rollout.py -v
```

On a TPU host: `scripts/accelerators/tpu/selfcr_smoke.py`, then the 0.6B recipe; the job log shows the per-sync
line `TPU colocated weight sync (raiden) for step N took ...`.

## Related Documentation

- [User Guide](./README.md)
- [vLLM Rollout](./rollout.md)
- [Weight Sync](./weight_sync.md)
