"""ElasticTrainer: FullyAsyncTrainer + attach/detach/joiner-sync under one lock + metrics tap.

Stock seams used (verl d4701e4e):
- ``FullyAsyncTrainer._setup_checkpoint_manager`` (fully_async_trainer.py): the ONLY place the
  standalone ``CheckpointEngineManager`` is built. Overridden to build ``LockedCheckpointManager``.
- ``CheckpointEngineManager.{replicas, add_replicas, remove_replicas, update_weights}``
  (checkpoint_engine/base.py). ``update_weights`` rebuilds the rollout worker group from
  ``self.replicas`` on every call, hence the lock.
- ``FullyAsyncTrainer._fit_update_weights`` calls ``await self.checkpoint_manager.update_weights(
  global_steps=self.current_param_version)`` after ``_fit_update_local_step`` bumped the version.
  Not overridden; the lock lives inside the manager subclass.
- ``self.logger`` (verl.utils.tracking.Tracking): every ``logger.log(data, step)`` is tapped (metrics.py).
  The per-sync-cycle aggregated metrics (``_fit_log_aggregated_training_metrics``) and the
  rollouter's reset_staleness timing (``_fit_update_weights``) are what we read; nothing is
  computed differently.

Observation-only overrides (no algorithmic change): ``_get_samples_from_queue`` (sets a
"waiting now" flag around ``super()``) and ``_fit_update_actor`` (marks actor weights dirty after
``super()``). The dirty flag drives ``joiner_sync_mode=clean_only``: a joiner-only sync is only
performed while the actor weights equal the last synced version; otherwise the (already attached)
joiner is covered by the next regular sync and the control side waits on the version gate.
"""

from __future__ import annotations

import itertools
import logging
import os
import time
from typing import Any, Callable

import ray

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _StockTrainer
from verl.utils.config import omega_conf_to_dataclass

from rl_pool_autoscaler._compat import rl_pool_config, unwrap_actor_class
from rl_pool_autoscaler.membership import MembershipRegistry, attach, default_key, detach, locked_update_weights, sync_joiners
from rl_pool_autoscaler.metrics import MetricsTap

logger = logging.getLogger(__name__)

_BaseTrainer = unwrap_actor_class(_StockTrainer)


# ---------------------------------------------------------------- NCCL group rotation (measured need, 2026-10-15)
# verl's NCCL checkpoint engine (nccl_checkpoint_engine.py) keeps ONE ray.util.collective group per worker under a
# fixed name. With rebuild_group=False the group of the launch-time membership is cached and any sync with a
# different world size fails (run m2-pods1b: joiner sync NCCL_ERROR_REMOTE_ERROR; the next regular sync would assert
# on world_size). With rebuild_group=True the group is destroyed in finalize and re-created under the SAME name at the
# next sync, which died with NCCL_ERROR_UNHANDLED_CUDA_ERROR on every trainer rank (run m2-pods1c). Workaround that
# touches no verl source: before each sync give every participating worker a fresh group name (the stock
# execute_checkpoint_engine RPC can call any engine attribute, including __setattr__), and destroy the PREVIOUS group
# one sync later under its own old name (no-op on workers that were not in it). rebuild_group stays False.
_GROUP_SEQ = itertools.count()
_GROUP_STATE: dict[str, Any] = {"prev": None, "events": []}
_DESTROY_PREV = os.environ.get("RL_POOL_NCCL_DESTROY_PREV", "1") not in ("0", "false", "False")


def _wg_set(wg, attr: str, value: Any) -> None:
    ws = wg.world_size
    ray.get(wg.execute_checkpoint_engine(["__setattr__"] * ws, [attr] * ws, [value] * ws))


def rotate_collective_group(actor_wg, rollout, destroy_prev: bool | None = None) -> str:
    """Give this sync a unique group name on all participants; destroy the previous group first (deferred)."""
    destroy_prev = _DESTROY_PREV if destroy_prev is None else destroy_prev
    name = f"rlpool_{os.getpid()}_{next(_GROUP_SEQ)}"
    prev = _GROUP_STATE["prev"]
    t0 = time.time()
    for wg in (actor_wg, rollout):
        if prev and destroy_prev:
            # finalize() destroys self.group_name when rebuild_group is True and rank >= 0; a worker that was not in
            # the previous group gets a harmless "group does not exist" warning from ray.util.collective.
            _wg_set(wg, "group_name", prev)
            _wg_set(wg, "rebuild_group", True)
            _wg_set(wg, "rank", 0)
            ray.get(wg.execute_checkpoint_engine(["finalize"] * wg.world_size))
            _wg_set(wg, "rebuild_group", False)
        _wg_set(wg, "group_name", name)
    _GROUP_STATE["prev"] = name
    ev = {"t": t0, "group": name, "destroyed": prev if destroy_prev else None, "rotate_s": time.time() - t0, "actor_ws": actor_wg.world_size, "rollout_ws": rollout.world_size}
    _GROUP_STATE["events"].append(ev)
    if len(_GROUP_STATE["events"]) > 200:
        del _GROUP_STATE["events"][:-200]
    print(f"[rl_pool] nccl group rotate {ev}", flush=True)
    return name


class RotatingGroupManager(CheckpointEngineManager):
    """CheckpointEngineManager that rotates the NCCL collective group name per sync (see above)."""

    def build_process_group(self, rollout):
        if getattr(self, "backend", None) == "nccl":
            rotate_collective_group(self.actor_wg, rollout)
        return super().build_process_group(rollout)


class LockedCheckpointManager(RotatingGroupManager):
    """CheckpointEngineManager whose update_weights holds the membership lock and records versions."""

    def __init__(self, config, actor_wg, replicas, registry: MembershipRegistry, on_synced: Callable[[int], None] | None = None, pre_sync=None):
        super().__init__(config=config, actor_wg=actor_wg, replicas=replicas)
        self._registry = registry
        self._on_synced = on_synced
        self._pre_sync = pre_sync  # async callable(manager) -> list of evicted ids; runs under the lock before every regular sync
        for r in replicas:
            rid = default_key(r)
            if rid not in registry.members:
                registry.members.append(rid)

    async def update_weights(self, global_steps: int = None):  # noqa: D401 - same signature as stock
        async def _stock():
            if self._pre_sync is not None:
                try:
                    evicted = await self._pre_sync(self)
                    if evicted:
                        print(f"[rl_pool] pre-sync sweep evicted dead replicas {evicted}", flush=True)
                except Exception as e:  # noqa: BLE001 - never block the sync on the sweep itself
                    print(f"[rl_pool] pre-sync sweep failed: {type(e).__name__}: {e}", flush=True)
            # Must be awaited from a coroutine frame: verl's @auto_await inspects the caller.
            return await CheckpointEngineManager.update_weights(self, global_steps=global_steps)

        result = await locked_update_weights(self._registry, self, int(global_steps or 0), kind="regular", do_sync=_stock)
        if self._on_synced is not None:
            self._on_synced(int(global_steps or 0))
        return result


class ElasticTrainer(_BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.membership = MembershipRegistry()
        self._pool_cfg = rl_pool_config(self.config)
        self._control_actor = None
        self._ckpt_engine_config = None
        self._weights_dirty = True  # until the first sync completes
        self._waiting_now = False
        self._wait_started_at: float | None = None
        self._metrics_tap = MetricsTap(self.logger, on_log=self._push)
        self.logger = self._metrics_tap  # stock code keeps calling self.logger.log(...)

    # ------------------------------------------------------------------ wiring
    def set_control_actor(self, handle) -> None:
        self._control_actor = handle

    def _push(self, snap: dict[str, Any] | None = None) -> None:
        if self._control_actor is None:
            return
        try:
            self._control_actor.push_trainer_metrics.remote(self.get_control_metrics(snap))
        except Exception as e:  # noqa: BLE001
            logger.warning("push_trainer_metrics failed: %s", e)

    def _on_regular_sync(self, global_steps: int) -> None:
        self._weights_dirty = False
        self._push()

    async def _setup_checkpoint_manager(self):
        replicas = await self.rollouter.get_replicas.remote()
        self._ckpt_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = LockedCheckpointManager(
            config=self._ckpt_engine_config,
            actor_wg=self.actor_wg,
            replicas=replicas,
            registry=self.membership,
            on_synced=self._on_regular_sync,
            pre_sync=self._pre_sync_sweep,
        )
        print(
            f"[ElasticTrainer] Locked checkpoint manager initialized (backend={self._ckpt_engine_config.backend}, "
            f"members={self.membership.members})"
        )

    async def _pre_sync_sweep(self, manager) -> list[str]:
        """Liveness check of every member right before a regular sync (lock already held by the caller):
        a replica whose actors died since the last 15 s poller pass would otherwise poison the NCCL
        group build (measured on TPU-M2; the GPU poller window is 15 s). Dead members are cut from
        the LB and the membership here (no lock re-entry: the registry is edited directly) and
        destroyed by the rollouter; the control actor's reconciler then replaces them."""
        health = await self.rollouter.check_health.remote(10.0)
        dead = [d for d in (health or {}).get("dead") or [] if d in self.membership.members]
        if not dead:
            return []
        await self.rollouter.evict_standalone.remote(dead)
        wanted = set(dead)
        victims = [r for r in list(manager.replicas) if default_key(r) in wanted]
        if victims:
            manager.remove_replicas(victims)
        self.membership.members = [rid for rid in self.membership.members if rid not in wanted]
        try:
            await self.rollouter.destroy_standalone.remote(dead)
        except Exception as e:  # noqa: BLE001
            logger.warning("destroy after pre-sync sweep failed: %s", e)
        self._push()
        return dead

    # ------------------------------------------------------------------ observation-only overrides
    async def _get_samples_from_queue(self):
        self._waiting_now = True
        self._wait_started_at = time.time()
        self._push()
        try:
            return await super()._get_samples_from_queue()
        finally:
            self._waiting_now = False
            self._wait_started_at = None
            self._push()

    def _fit_update_actor(self, batch):
        out = super()._fit_update_actor(batch)
        self._weights_dirty = True
        return out

    # ------------------------------------------------------------------ scaling lever
    async def attach_replicas(self, replicas: list[Any]) -> list[str]:
        """Add replicas to the checkpoint membership (under the lock). Accepts ids or replica objects."""
        ids = [r for r in replicas if isinstance(r, str)]
        objs = [r for r in replicas if not isinstance(r, str)]
        if ids:
            objs += await self.rollouter.get_standalone_replica_objects.remote(ids)
        added = await attach(self.membership, self.checkpoint_manager, objs)
        self._push()
        return added

    async def detach_replicas(self, replicas: list[Any]) -> list[str]:
        ids = [r if isinstance(r, str) else default_key(r) for r in replicas]
        removed = await detach(self.membership, self.checkpoint_manager, ids)
        self._push()
        return removed

    async def sync_joiners_now(self, replica_ids: list[str], global_steps: int | None = None, mode: str | None = None) -> dict[str, Any]:
        """Joiner-only weight sync through a temporary CheckpointEngineManager over just the joiners.

        Returns {"version", "deferred", "reason"}. Deferred when ``mode == "clean_only"`` and the
        actor weights have moved since the last regular sync (the attached joiner will be covered by
        the next regular sync; the caller waits on the version gate).
        """
        mode = mode or self._pool_cfg.joiner_sync_mode
        gs = self.current_param_version if global_steps is None else int(global_steps)
        ids = [r if isinstance(r, str) else default_key(r) for r in replica_ids]
        if mode == "clean_only" and self._weights_dirty:
            return {
                "version": min((self.membership.versions.get(rid, -1) for rid in ids), default=-1),
                "deferred": True,
                "reason": f"actor weights dirty (local_trigger_step={self.local_trigger_step}); waiting for regular sync",
                "last_synced_version": self.membership.last_synced_version,
            }

        def make_manager(reps):
            return RotatingGroupManager(config=self._ckpt_engine_config, actor_wg=self.actor_wg, replicas=reps)

        t0 = time.time()
        version = await sync_joiners(self.membership, make_manager, self.checkpoint_manager, ids, gs)
        self._push()
        return {"version": version, "deferred": False, "reason": "synced", "wall_s": time.time() - t0, "last_synced_version": self.membership.last_synced_version}

    # ------------------------------------------------------------------ queries
    def get_membership(self) -> dict[str, Any]:
        snap = self.membership.snapshot()
        snap["nccl_groups"] = list(_GROUP_STATE["events"][-5:])
        return snap

    def get_control_metrics(self, snap: dict[str, Any] | None = None) -> dict[str, Any]:
        snap = dict(snap) if snap is not None else self._metrics_tap.control_snapshot()
        snap.update(
            {
                "current_param_version": int(self.current_param_version),
                "local_trigger_step": int(self.local_trigger_step),
                "global_steps": int(self.global_steps),
                "weights_clean": not self._weights_dirty,
                "trainer_waiting_now": self._waiting_now,
                "trainer_wait_current_s": (time.time() - self._wait_started_at) if self._wait_started_at else 0.0,
                "membership": self.membership.snapshot(),
                "t": time.time(),
            }
        )
        return snap
