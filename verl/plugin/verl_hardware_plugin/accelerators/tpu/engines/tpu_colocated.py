# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Checkpoint engine manager for the TPU colocated mode: trainer and rollout alternate on one slice.

verl's V1 trainers with hybrid vLLM replicas on the trainer's own workers (``hybrid_engine=True``)
drive this manager through the stock interface: the default sync trainer calls ``update_weights`` at
init and at the end of every step and ``sleep_replicas`` after sampling; ``colocate_async`` calls the
same two plus ``abort_replicas`` before the sleep (partial rollout) and ``resume_generation_replicas``
after ``update_weights``. The manager is neutral to the mode: it never pauses generation itself (its
``abort_replicas`` is a no-op while the rollout holds no chips), so ``update_weights`` has nothing to
resume and leaves that to the trainer that paused. On GPU the stock manager only sleeps and wakes vLLM,
because one CUDA device can be open to the trainer and to vLLM at once. A TPU chip has one owner process, so this subclass composes three concerns that
each keep their single meaning:

1. memory: vLLM ``sleep(level 2)`` / ``wake_up(tags)`` on the rollout (``rollout/tpu_sleep.py`` until
   vllm-torchtpu ships it), issued through the same server calls the GPU loop uses;
2. the tpu-sync receive buffers: ``drop_receive_buffers`` / ``rebind_receive_buffers`` on the Raiden
   worker extension, through ``collective_rpc``;
3. the device: ``platform_tpu.suspend_device`` / ``resume_device``, on the rollout through the
   extension's ``suspend_device`` / ``resume_device`` and on the trainer through
   ``worker.__ray_call__`` on every trainer worker actor (the Ray facility upstream itself uses to
   read the workers' node ids). No worker RPC and no platform hook are added.

Sequence (``T`` = trainer workers, ``S`` = rollout workers):

- ``sleep_replicas``: S vLLM sleep(2) -> S drop_receive_buffers -> S suspend_device -> T resume_device
- ``update_weights``: T stage + transfer (T resident) -> T release send buffers -> T suspend_device ->
  S resume_device -> S wake_up(["weights"]) -> S rebind_receive_buffers -> S install ->
  S wake_up(["kv_cache"]) -> set_global_steps. The first sync inserts one round trip
  so the rollout workers can register their receive buffers (sized from the shapes the trainer publishes
  in its first stage) while they hold the chips.

Invariant: at any instant at most one of the two process families holds the chips. Selected with
``actor_rollout_ref.rollout.checkpoint_manager_class=<this module>.TPUColocatedCheckpointEngineManager``
on a verl that honours the key; otherwise ``select_checkpoint_manager`` is bound into the V1 trainer
module by ``TPUvLLMReplica.init_hybrid`` and picks this class for TPU hybrid replicas (see
``install_manager_selector``).

Requires ``actor.param_offload=false`` / ``optimizer_offload=false`` (the engine's ``to("cpu")`` tensor
offload stays unused; the device release here is the only hand-over) and ``rollout.free_cache_engine=true``
(upstream gates the hybrid sleep and the KV wake on it).
"""

import asyncio
import logging
import os
import sys
import time
from typing import Any, Callable

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.utils.ray_utils import auto_await
from verl_hardware_plugin.accelerators.tpu import platform_tpu
from verl_hardware_plugin.accelerators.tpu.engines.raiden_checkpoint_engine import (
    _as_bool,
    _get_or_start_controller,
    _raiden_engine_kwargs,
    raiden_install,
    raiden_sampler_init,
    raiden_trainer_stage,
    raiden_transfer,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

COLOCATED_BACKENDS = ("raiden", "tpu")
TRAINER_BASE_MODULE = "verl.trainer.ppo.v1.trainer_base"


def _suspend_trainer_worker(worker_self: Any) -> dict:
    """Runs inside a trainer worker actor through ``__ray_call__``."""
    return platform_tpu.suspend_device()


def _resume_trainer_worker(worker_self: Any) -> dict:
    """Runs inside a trainer worker actor through ``__ray_call__``."""
    return platform_tpu.resume_device()


def _max_numeric(results: Any) -> dict[str, float]:
    """Max over workers of every numeric entry in the (nested) per-worker dicts a call returned."""
    out: dict[str, float] = {}
    stack = list(results) if isinstance(results, list | tuple) else [results]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    out[key] = max(out.get(key, value), value)
        elif isinstance(item, list | tuple):
            stack.extend(item)
    return out


def configured_backend(replicas: list) -> str | None:
    """The checkpoint-engine backend of the run config, read from the replicas' rollout config.

    ``PPOTrainer._setup`` overwrites the backend of the config it hands to the manager with ``naive`` for
    the stock manager class; the ``RolloutConfig`` the replicas hold still carries the configured value.
    """
    for replica in replicas or []:
        backend = getattr(getattr(getattr(replica, "config", None), "checkpoint_engine", None), "backend", None)
        if backend:
            return backend
    return None


def is_hybrid_replica(replica: Any) -> bool:
    mode = getattr(replica, "rollout_mode", None)
    return getattr(mode, "value", mode) == "hybrid"


class TPUColocatedCheckpointEngineManager(CheckpointEngineManager):
    """``CheckpointEngineManager`` whose hand-overs alternate the trainer and the rollout on one TPU slice."""

    def __init__(self, config, actor_wg, replicas):
        real_backend = configured_backend(replicas)
        if config.backend == "naive" and real_backend in COLOCATED_BACKENDS:
            # The V1 trainer forces "naive" for the stock manager; the run config says what moves the bytes.
            config.backend = real_backend
        if config.backend not in COLOCATED_BACKENDS:
            raise ValueError(
                f"TPUColocatedCheckpointEngineManager needs checkpoint_engine.backend in {COLOCATED_BACKENDS}, "
                f"got {config.backend!r}"
            )
        super().__init__(config=config, actor_wg=actor_wg, replicas=replicas)
        # TPUvLLMReplica.init_hybrid released the trainer's chips so the rollout could compile; the rollout
        # holds them when this manager is built.
        self._trainer_suspended = bool(replicas)
        self._samplers_suspended = False
        self._samplers_registered = False
        self._pending: dict[str, float] = {}
        self.last_sync_metrics: dict[str, float] = {}
        logger.info(
            f"TPU colocated manager ({self.backend} weight sync): {len(replicas)} hybrid replica(s) on the "
            f"trainer's {actor_wg.world_size} chips"
        )

    # ------------------------------------------------------------------
    # The three concerns
    # ------------------------------------------------------------------

    async def _trainer_call(self, fn: Callable, label: str) -> list:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[worker.__ray_call__.remote(fn) for worker in self.actor_wg.workers])
        self._pending[f"timing_s/tpu-colocated/{label}"] = time.perf_counter() - t0
        self._log_sizes(f"trainer {label}", results)
        return results

    async def _rollout_rpc(self, method: str, label: str, **kwargs) -> list:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[replica.server_handle.collective_rpc.remote(method=method, **kwargs) for replica in self.replicas]
        )
        self._pending[f"timing_s/tpu-colocated/{label}"] = time.perf_counter() - t0
        self._log_sizes(f"rollout {method}", results)
        return results

    async def _wake(self, tags: list[str], label: str) -> None:
        t0 = time.perf_counter()
        await asyncio.gather(*[server.wake_up.remote(tags=tags) for r in self.replicas for server in r.servers])
        self._pending[f"timing_s/tpu-colocated/{label}"] = time.perf_counter() - t0

    def _log_sizes(self, tag: str, results: Any) -> None:
        sizes = {k: v for k, v in _max_numeric(results).items() if k.endswith("_mib") or k.endswith("_s")}
        if sizes:
            logger.info(f"{tag}: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(sizes.items())))

    async def _suspend_trainer(self) -> None:
        if not self._trainer_suspended:
            await self._trainer_call(_suspend_trainer_worker, "trainer_suspend")
            self._trainer_suspended = True

    async def _resume_trainer(self) -> None:
        if self._trainer_suspended:
            await self._trainer_call(_resume_trainer_worker, "trainer_resume")
            self._trainer_suspended = False

    async def _suspend_samplers(self) -> None:
        if not self._samplers_suspended:
            await self._rollout_rpc("drop_receive_buffers", "sampler_drop")
            await self._rollout_rpc("suspend_device", "sampler_suspend")
            self._samplers_suspended = True

    async def _resume_samplers(self) -> None:
        if self._samplers_suspended:
            await self._rollout_rpc("resume_device", "sampler_resume")
            self._samplers_suspended = False

    # ------------------------------------------------------------------
    # The manager interface the trainer loop calls
    # ------------------------------------------------------------------

    @auto_await
    async def sleep_replicas(self):
        """After sampling: free the rollout's memory, release its chips, give them back to the trainer."""
        if not self._samplers_suspended:
            t0 = time.perf_counter()
            await super().sleep_replicas()  # drain + vLLM sleep(level 2): KV and weight storage freed
            self._pending["timing_s/tpu-colocated/sampler_sleep"] = time.perf_counter() - t0
            await self._suspend_samplers()
        await self._resume_trainer()

    @auto_await
    async def wake_up_replicas(self):
        """Give the chips to the rollout and wake vLLM; the loop does not call this, update_weights does more."""
        await self._suspend_trainer()
        await self._resume_samplers()
        await self._wake(["weights", "kv_cache"], "wake")
        await self._rollout_rpc("rebind_receive_buffers", "sampler_rebind")

    @auto_await
    async def abort_replicas(self, reject_request: bool = False):
        # Nothing is in flight while the rollout holds no chips (it was drained before the release), and
        # aborting needs a live engine. The GPU naive path does not abort in update_weights either.
        if self._samplers_suspended:
            logger.debug("abort_replicas skipped: rollout workers hold no chips")
            return
        await super().abort_replicas(reject_request=reject_request)

    @auto_await
    async def update_weights(self, global_steps: int = None):
        """End of step: move the weights while the trainer holds the chips, hand them to the rollout, install."""
        t_start = time.perf_counter()
        metrics = dict(self._pending)
        self._pending = {}
        step_key = global_steps if global_steps is not None else 0

        # The trainer must hold the chips for the stage (first call after init: it was released for the
        # rollout's compile and the loop's setup sleep_replicas already brought it back; keep it robust).
        if self._trainer_suspended:
            await self.sleep_replicas()
            metrics.update(self._pending)
            self._pending = {}

        if self.backend == "raiden":
            verify_parity = _as_bool(_raiden_engine_kwargs(self.config).get("verify_parity", False))
            _get_or_start_controller(self)
            if not self._samplers_registered:
                # First sync: the rollout workers size and register their receive buffers from the global
                # shapes the trainer publishes in its first stage, and need their chips for the staging
                # allocation. One extra round trip, once.
                metrics["timing_s/tpu-sync/trainer_init_first"] = await raiden_trainer_stage(self, global_steps)
                await self._suspend_trainer()
                await self._resume_samplers()
                metrics["timing_s/tpu-sync/sampler_init"] = await raiden_sampler_init(self)
                self._samplers_registered = True
                await self._suspend_samplers()
                await self._resume_trainer()
            metrics["timing_s/tpu-sync/trainer_init"] = await raiden_trainer_stage(self, global_steps)
            transfer = await raiden_transfer(self, global_steps)  # release awaited: the trainer leaves next
            metrics["timing_s/tpu-sync/barrier"] = transfer["barrier"]
            metrics["timing_s/tpu-sync/p2p_transfer"] = transfer["transfer"]
            metrics["timing_s/tpu-sync/trainer_release"] = transfer["release"]
            await self._suspend_trainer()
            await self._resume_samplers()
            await self._wake(["weights"], "wake_weights")
            await self._rollout_rpc("rebind_receive_buffers", "sampler_rebind")
            install = await raiden_install(self, global_steps, verify_parity=verify_parity)
            metrics["timing_s/tpu-sync/sampler_h2d"] = install["install"]
            if install.get("h2d_pure") is not None:
                metrics["timing_s/tpu-sync/sampler_h2d_pure"] = install["h2d_pure"]
        else:
            # Plasma "tpu" backend: rank 0 publishes the packed weights while resident, every rollout
            # worker pulls its shard after the hand-over.
            t0 = time.perf_counter()
            actor_refs = self.actor_wg.update_weights(global_steps=global_steps, mode=self.backend)
            if actor_refs is not None:
                import ray

                await asyncio.to_thread(ray.get, actor_refs)
            metrics["timing_s/tpu-sync/trainer_init"] = time.perf_counter() - t0
            await self._suspend_trainer()
            await self._resume_samplers()
            await self._wake(["weights"], "wake_weights")
            t0 = time.perf_counter()
            await self._rollout_rpc("load_weights_from_ray_registry", "sampler_load", args=(step_key,))
            metrics["timing_s/tpu-sync/sampler_h2d"] = time.perf_counter() - t0
            if global_steps is not None:
                await asyncio.gather(
                    *[replica.server_handle.set_global_steps.remote(global_steps) for replica in self.replicas]
                )

        t0 = time.perf_counter()
        await self.resume_kv_cache_replicas()  # wake_up(["kv_cache"]) + prefix-cache reset on every server
        metrics["timing_s/tpu-colocated/wake_kv"] = time.perf_counter() - t0
        metrics.update(self._pending)
        self._pending = {}
        metrics["timing_s/tpu-sync/total_sync"] = time.perf_counter() - t_start
        self.last_sync_metrics = metrics
        parts = [f"{k.rsplit('/', 1)[1]} {v:.2f}s" for k, v in metrics.items() if k != "timing_s/tpu-sync/total_sync"]
        logger.info(
            f"TPU colocated weight sync ({self.backend}) for step {global_steps} took "
            f"{metrics['timing_s/tpu-sync/total_sync']:.3f}s ({' | '.join(parts)})"
        )
        # No resume_generation_replicas() here: this manager never paused generation (the stock
        # manager resumes inside update_weights because its own step 1 aborted). The sync trainer
        # never pauses; colocate_async aborted on_sample_end and resumes after update_weights itself.
        return metrics


# ----------------------------------------------------------------------
# Selection without the verl-main change (C1)
# ----------------------------------------------------------------------


def select_checkpoint_manager(config, actor_wg, replicas) -> CheckpointEngineManager:
    """Stand-in for the ``CheckpointEngineManager`` name in the V1 trainer module.

    Picks ``TPUColocatedCheckpointEngineManager`` when the platform is TPU, the replicas are hybrid
    (``colocate_async`` or any trainer with hybrid replicas) and the configured backend is ``raiden`` or
    ``tpu``; the stock manager otherwise. Same subclass, same behaviour as selecting it through
    ``rollout.checkpoint_manager_class``, which retires this once verl main resolves that key.
    """
    from verl.plugin.platform import get_platform

    on_tpu = get_platform().device_name == "tpu"
    hybrid = bool(replicas) and all(is_hybrid_replica(r) for r in replicas)
    backend = configured_backend(replicas)
    if on_tpu and hybrid and backend in COLOCATED_BACKENDS:
        logger.info(
            f"Selecting TPUColocatedCheckpointEngineManager for {len(replicas)} hybrid replica(s), {backend=}"
        )
        return TPUColocatedCheckpointEngineManager(config=config, actor_wg=actor_wg, replicas=replicas)
    return CheckpointEngineManager(config=config, actor_wg=actor_wg, replicas=replicas)


def install_manager_selector() -> bool:
    """Bind ``select_checkpoint_manager`` to the ``CheckpointEngineManager`` name in the V1 trainer module.

    One guarded assignment, done from ``TPUvLLMReplica.init_hybrid`` (the trainer module is imported by
    then, and that is the only path that leads to the colocated manager). Returns True when installed.
    """
    trainer_base = sys.modules.get(TRAINER_BASE_MODULE)
    if trainer_base is None:
        return False
    current = getattr(trainer_base, "CheckpointEngineManager", None)
    if current is select_checkpoint_manager:
        return True
    if current is not CheckpointEngineManager:
        logger.info(f"{TRAINER_BASE_MODULE}.CheckpointEngineManager is {current!r}; leaving it alone")
        return False
    trainer_base.CheckpointEngineManager = select_checkpoint_manager
    logger.info(f"Bound select_checkpoint_manager to {TRAINER_BASE_MODULE}.CheckpointEngineManager (TPU hybrid)")
    return True
