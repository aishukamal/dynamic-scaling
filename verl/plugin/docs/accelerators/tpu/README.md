# VERL Google TPU User Guide

## Introduction

This document describes Google TPU support in `verl-hardware-plugin`.

The plugin registers the TPU platform (`PlatformTPU`: device metadata, Ray resource configuration,
and the PJRT worker environment), the TorchTitan TPU training engine (`TorchTitanTPUEngineWithLMHead`
under `verl_hardware_plugin/accelerators/tpu/engines/torchtitan_tpu.py`), a vLLM rollout for TPU, and two checkpoint
engines that send the trainer's weights to it (`tpu` and `raiden`).

## Directory Structure

```text
verl_hardware_plugin/accelerators/tpu/
├── platform_tpu.py                  # TPU platform settings, suspend_device / resume_device
├── tpu_control.py                   # libtpu control-pipe client (in-process suspend / resume)
├── engines/
│   ├── raiden_checkpoint_engine.py   # Peer-to-peer weight sync to the vLLM rollout over tpu-sync (Raiden)
│   ├── ray_weight_registry.py        # Ray actor where trainer and rollout exchange weight-sync metadata
│   ├── torchtitan_tpu.py             # TorchTitan TPU training engine (FSDP2 / SPMD)
│   ├── tpu_checkpoint_engine.py      # Sends the trainer's weights to the vLLM rollout
│   ├── tpu_colocated.py              # Checkpoint engine manager for trainer and rollout on one slice
│   └── tpu_utils.py                  # Sequence bucketing and TorchTitan TPU config/input helpers
└── rollout/
    ├── tpu_raiden.py                 # vLLM worker extension that receives Raiden weight syncs
    ├── tpu_sleep.py                  # Memory-only sleep / wake_up for vllm-torchtpu's TPUWorker
    ├── tpu_vllm.py                   # vLLM rollout replica and server for TPU
    └── tpu_vllm_patches.py           # vLLM / vllm-torchtpu patches for multi-host TPU
```

```text
docs/accelerators/tpu/
├── README.md                         # This file
├── colocated.md                      # Trainer and rollout on one slice (verl's sync trainer with hybrid replicas on TPU)
├── install_guidance.md               # Installation and environment setup
├── quick_start.md                    # Selecting and verifying the platform
├── rollout.md                        # Running the vLLM rollout on TPU
└── weight_sync.md                    # Choosing and configuring the weight sync (tpu / raiden)
```

## Getting Started

- [Installation Guide](./install_guidance.md) — prerequisites and environment setup
- [Quick Start](./quick_start.md) — select the TPU platform and verify it resolves
- [vLLM Rollout](./rollout.md) — run GRPO with the vLLM rollout on TPU
- [Weight Sync](./weight_sync.md) — send the trainer's weights to the rollout peer to peer with `raiden`
- [Colocated Mode](./colocated.md) — trainer and rollout alternating on one slice with verl's default sync trainer (colocate_async also works)

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `tpu` |
| Vendor identifier | `google` |
| Communication backend | `tpu_dist` (registered by `torch_tpu`) |
| Device visibility env var | `CUDA_VISIBLE_DEVICES` (see note below) |
| Ray resource name | `TPU` |
| IPC support | No |
| Colocated worker groups | Not supported — a chip belongs to a single process; trainer and rollout share a slice through the [colocated mode](./colocated.md) instead |

### Why the visibility env var is `CUDA_VISIBLE_DEVICES`

This is intentional. verl assigns `os.environ[<this key>]` when launching vLLM servers. Pointing it
at `TPU_VISIBLE_CHIPS` would overwrite the per-worker chip index that the platform sets through
`get_worker_env_vars()` and reads back through `ray_local_rank_override()`, breaking rank mapping.
The two variables serve different purposes and must not be merged.

## Chip Support

| Generation | HBM per chip | Topologies with a built-in mapping |
|------------|--------------|------------------------------------|
| v6e | 32 GB | `v6e-4`, `v6e-8`, `v6e-16`, `v6e-32`, `v6e-64` |
| v5e | 16 GB (not in the HBM table yet) | `v5litepod-4`, `v5litepod-8`, `v5litepod-16`, `v5litepod-32`, `v5litepod-64` |

When the generation cannot be determined from Ray node labels or environment variables, the platform
falls back to the v6e HBM figure. `VERL_TPU_TOPOLOGY_<n>` overrides the topology of an `n`-chip pool.

## Related Documentation

- [verl plugin system](../../development.md)
