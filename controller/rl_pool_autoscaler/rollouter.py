"""ElasticRollouter: FullyAsyncRollouter + a parked/active standalone replica pool.

Stock seams used (verl d4701e4e + cherry-pick 976b5f57):
- ``FullyAsyncLLMServerManager`` (fully_async_rollouter.py): ``rollout_replicas`` (= ACTIVE standalone
  replicas; what ``get_replicas()`` returns to the trainer), ``server_handles`` / ``server_addresses``,
  ``global_load_balancer`` (GlobalRequestLoadBalancer actor: ``add_servers`` / ``remove_servers`` /
  ``get_status``), ``rollout_replica_class``, ``rollout_config``, ``model_config``,
  ``prometheus_server_addresses``.
- ``RolloutReplica.init_standalone`` (replica.py): creates its own ResourcePoolManager + placement
  group (named ``rollout_pool_{rank}`` -> steered by ``ray_pg_extra_resources.rollout_pool``) and
  worker group, then launches the server actors.
- ``RolloutReplica.{sleep,wake_up,abort_all_requests,resume_generation}``.
  NOTE: for vLLM in STANDALONE mode ``sleep``/``wake_up``/``release_kv_cache`` are NO-OPS at this
  commit (vllm_async_server.py: "skip sleep in standalone mode"). "Parked" therefore means: out of
  the LB, out of the trainer's checkpoint membership, engine paused (abort leaves it paused), GPU
  memory still held (warm). Real memory release is only ``destroy_standalone`` (release=True).
- ``FullyAsyncRollouter._update_max_concurrent_samples`` after every LB membership change.
- ``verl.workers.rollout.utils.update_prometheus_config`` when ``rollout.prometheus.enable``.

States of a standalone replica (tracked here by id ``standalone_{replica_rank}``):
    ACTIVE  in ``rollout_replicas`` + LB + trainer membership (trainer side tracks membership)
    PARKED  in ``self._parked``; not in LB; not in ``rollout_replicas``; engine paused
    (gone)  destroyed: actors killed, placement group removed

Transitions are driven by the control actor through the ScaleOrchestrator; this class only
exposes the primitive operations and keeps the three stock lists consistent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import ray
from omegaconf import open_dict

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter as _StockRollouter
from verl.workers.rollout.replica import RolloutReplica
from verl.workers.rollout.utils import update_prometheus_config

from rl_pool_autoscaler._compat import unwrap_actor_class

logger = logging.getLogger(__name__)

_BaseRollouter = unwrap_actor_class(_StockRollouter)


def replica_id(replica: RolloutReplica) -> str:
    return f"standalone_{replica.replica_rank}"


class ElasticRollouter(_BaseRollouter):
    def __init__(self, config, tokenizer, processor=None, device_name=None):
        super().__init__(config=config, tokenizer=tokenizer, processor=processor, device_name=device_name)
        self._parked: dict[str, RolloutReplica] = {}
        self._next_replica_rank: int | None = None
        self._pool_lock: asyncio.Lock | None = None
        self._pool_events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ plumbing
    def _init_async_objects(self):
        super()._init_async_objects()
        self._pool_lock = asyncio.Lock()

    @property
    def _mgr(self):
        return self.llm_server_manager

    def _record(self, action: str, **detail: Any) -> None:
        ev = {"t": time.time(), "action": action, **detail}
        self._pool_events.append(ev)
        if len(self._pool_events) > 500:
            del self._pool_events[:-500]
        print(f"[ElasticRollouter] {action} {detail}", flush=True)

    def _alloc_rank(self) -> int:
        if self._next_replica_rank is None:
            ranks = [r.replica_rank for r in self._mgr.rollout_replicas]
            ranks += [r.replica_rank for r in self._mgr.hybrid_replicas.values()]
            ranks += [r.replica_rank for r in self._parked.values()]
            self._next_replica_rank = (max(ranks) + 1) if ranks else 0
        rank = self._next_replica_rank
        self._next_replica_rank += 1
        return rank

    def _find(self, rid: str) -> tuple[RolloutReplica | None, str]:
        if rid in self._parked:
            return self._parked[rid], "parked"
        for r in self._mgr.rollout_replicas:
            if replica_id(r) == rid:
                return r, "active"
        return None, "unknown"

    def _refresh_prometheus(self) -> None:
        rc = self._mgr.rollout_config
        if not rc.prometheus.enable or rc.disable_log_stats:
            return
        try:
            addresses = list(self._mgr.prometheus_server_addresses) + list(self._mgr.server_addresses)
            update_prometheus_config(rc.prometheus, addresses, rc.name)
        except Exception as e:  # noqa: BLE001
            logger.warning("prometheus refresh failed: %s", e)

    # ------------------------------------------------------------------ queries
    def get_active_standalone_ids(self) -> list[str]:
        return [replica_id(r) for r in self._mgr.rollout_replicas]

    def get_parked_standalone_ids(self) -> list[str]:
        return list(self._parked.keys())

    def get_standalone_replica_objects(self, replica_ids: list[str]) -> list[RolloutReplica]:
        """Replica objects for the trainer (pickled across the actor boundary; identity by replica_rank)."""
        out = []
        for rid in replica_ids:
            r, state = self._find(rid)
            if r is None:
                raise KeyError(f"unknown standalone replica {rid}")
            out.append(r)
        return out

    async def get_pool_status(self) -> dict[str, Any]:
        lb_status: dict[str, Any] = {}
        try:
            lb_status = await self._mgr.global_load_balancer.get_status.remote()
        except Exception as e:  # noqa: BLE001
            lb_status = {"error": str(e)}
        try:
            stats = await self.get_statistics()
        except Exception as e:  # noqa: BLE001
            stats = {"error": str(e)}
        inflight = lb_status.get("servers", {}) if isinstance(lb_status, dict) else {}
        active = [
            {"id": replica_id(r), "address": r._server_address, "inflight": inflight.get(r._server_address, 0)}
            for r in self._mgr.rollout_replicas
        ]
        parked = [{"id": rid, "address": r._server_address} for rid, r in self._parked.items()]
        return {
            "active": active,
            "parked": parked,
            "hybrid_active": len(self._mgr.alive_replicas),
            "max_concurrent_samples": self.max_concurrent_samples,
            "concurrent_samples_per_replica": self.concurrent_samples_per_replica,
            "max_queue_size": self.max_queue_size,
            "stats": stats,
            "lb_total_inflight": lb_status.get("total_inflight") if isinstance(lb_status, dict) else None,
            "paused": self.paused,
            "recent_events": self._pool_events[-10:],
        }

    async def check_health(self, timeout_s: float = 10.0) -> dict[str, Any]:
        """Liveness of every ACTIVE replica's server and worker actors (``__ray_ready__`` with a timeout).
        Returns {"dead": [ids], "checked": n, "errors": {id: msg}}. Used by the control actor's poller
        to evict a replica whose process died (the stock standalone path has no health handling)."""
        dead: list[str] = []
        errors: dict[str, str] = {}
        reps = list(self._mgr.rollout_replicas)
        for r in reps:
            rid = replica_id(r)
            handles = list(getattr(r, "servers", []) or []) + list(getattr(r, "workers", []) or [])
            for h in handles:
                try:
                    await asyncio.wait_for(h.__ray_ready__.remote(), timeout=timeout_s)
                except Exception as e:  # noqa: BLE001 - RayActorError, timeout, ...
                    dead.append(rid)
                    errors[rid] = f"{type(e).__name__}: {str(e)[:200]}"
                    break
        return {"dead": dead, "checked": len(reps), "errors": errors, "t": time.time()}

    async def evict_standalone(self, replica_ids: list[str]) -> list[str]:
        """Failure path: drop ACTIVE replicas from the LB and the three stock lists WITHOUT touching their
        (possibly dead) actors, park them, and let ``destroy_standalone`` clean up what is left."""
        async with self._pool_lock:
            wanted = set(replica_ids)
            reps = [r for r in self._mgr.rollout_replicas if replica_id(r) in wanted]
            if not reps:
                return []
            await self._lb_remove(reps)
            for r in reps:
                self._parked[replica_id(r)] = r
            self._refresh_prometheus()
            self._record("evict", ids=[replica_id(r) for r in reps], active=self.get_active_standalone_ids())
            return [replica_id(r) for r in reps]

    # ------------------------------------------------------------------ launch-time
    async def prestart_park(self, keep_active: int) -> dict[str, Any]:
        """Park all but the first ``keep_active`` standalone replicas.

        Must run after ``init_workers()`` and BEFORE ``set_max_required_samples()`` and before the
        trainer's ``set_rollouter()`` so that max_concurrent_samples and the trainer's initial
        checkpoint membership only see the active set. No requests are in flight yet.
        """
        async with self._pool_lock:
            active = list(self._mgr.rollout_replicas)
            victims = active[keep_active:]
            if not victims:
                return {"parked": [], "active": self.get_active_standalone_ids()}
            ids = [replica_id(r) for r in victims]
            await self._lb_remove(victims)
            await asyncio.gather(*[r.abort_all_requests() for r in victims])  # leaves engines paused
            await asyncio.gather(*[r.sleep() for r in victims])  # no-op for vLLM standalone; real on sleep-tier backends
            for r in victims:
                self._parked[replica_id(r)] = r
            self._refresh_prometheus()
            self._record("prestart_park", parked=ids, active=self.get_active_standalone_ids())
            return {"parked": ids, "active": self.get_active_standalone_ids()}

    # ------------------------------------------------------------------ scale-out primitives
    async def add_standalone(self, k: int) -> list[str]:
        """Create ``k`` NEW standalone replicas (each with its own placement group). They start PARKED
        (not in the LB) so the trainer can attach + sync them before they serve."""
        async with self._pool_lock:
            mgr = self._mgr
            rc = mgr.rollout_config
            standalone_gmu = rc.get("standalone_gpu_memory_utilization", None)
            original_gmu = rc.get("gpu_memory_utilization", None)
            override = standalone_gmu is not None and original_gmu is not None
            t0 = time.time()
            try:
                if override:
                    with open_dict(rc):
                        rc.gpu_memory_utilization = standalone_gmu
                # RolloutReplica.__init__ snapshots the config into a dataclass, so the override
                # must be in place at construction time (same trick as FullyAsyncLLMServerManager).
                new = [
                    mgr.rollout_replica_class(
                        replica_rank=self._alloc_rank(),
                        config=rc,
                        model_config=mgr.model_config,
                        gpus_per_node=rc.n_gpus_per_node,
                    )
                    for _ in range(int(k))
                ]
                await asyncio.gather(*[r.init_standalone() for r in new])
            finally:
                if override:
                    with open_dict(rc):
                        rc.gpu_memory_utilization = original_gmu
            ids = []
            for r in new:
                rid = replica_id(r)
                self._parked[rid] = r
                ids.append(rid)
            self._record("add_standalone", created=ids, create_s=time.time() - t0)
            return ids

    async def wake_standalone(self, replica_ids: list[str]) -> list[str]:
        reps = [self._parked[rid] for rid in replica_ids if rid in self._parked]
        await asyncio.gather(*[r.wake_up() for r in reps])
        self._record("wake", ids=[replica_id(r) for r in reps])
        return [replica_id(r) for r in reps]

    async def resume_standalone(self, replica_ids: list[str]) -> list[str]:
        reps = [self._find(rid)[0] for rid in replica_ids]
        reps = [r for r in reps if r is not None]
        await asyncio.gather(*[r.resume_generation() for r in reps])
        return [replica_id(r) for r in reps]

    async def lb_add(self, replica_ids: list[str]) -> int:
        """Register parked replicas in the load balancer and make them ACTIVE. Idempotent."""
        async with self._pool_lock:
            mgr = self._mgr
            reps = [self._parked[rid] for rid in replica_ids if rid in self._parked]
            if not reps:
                return 0
            servers = {r._server_address: r._server_handle for r in reps}
            await mgr.global_load_balancer.add_servers.remote(servers=servers)
            for r in reps:
                if r._server_address not in mgr.server_addresses:
                    mgr.server_handles.append(r._server_handle)
                    mgr.server_addresses.append(r._server_address)
                if r not in mgr.rollout_replicas:
                    mgr.rollout_replicas.append(r)
                self._parked.pop(replica_id(r), None)
            self._update_max_concurrent_samples()
            self._refresh_prometheus()
            self._record("lb_add", ids=[replica_id(r) for r in reps], active=self.get_active_standalone_ids())
            return len(reps)

    # ------------------------------------------------------------------ scale-in primitives
    async def _lb_remove(self, reps: list[RolloutReplica]) -> None:
        mgr = self._mgr
        await mgr.global_load_balancer.remove_servers.remote(server_ids=[r._server_address for r in reps])
        for r in reps:
            if r._server_address in mgr.server_addresses:
                idx = mgr.server_addresses.index(r._server_address)
                mgr.server_addresses.pop(idx)
                mgr.server_handles.pop(idx)
            if r in mgr.rollout_replicas:
                mgr.rollout_replicas.remove(r)
        self._update_max_concurrent_samples()

    async def lb_remove(self, replica_ids: list[str]) -> int:
        """Remove ACTIVE replicas from the load balancer and mark them PARKED. Idempotent.
        Does NOT abort: call ``abort_standalone`` next so in-flight requests re-route (partial rollout)."""
        async with self._pool_lock:
            wanted = set(replica_ids)
            reps = [r for r in self._mgr.rollout_replicas if replica_id(r) in wanted]
            if not reps:
                return 0
            await self._lb_remove(reps)
            for r in reps:
                self._parked[replica_id(r)] = r
            self._refresh_prometheus()
            self._record("lb_remove", ids=[replica_id(r) for r in reps], active=self.get_active_standalone_ids())
            return len(reps)

    async def abort_standalone(self, replica_ids: list[str]) -> list[str]:
        reps = [self._find(rid)[0] for rid in replica_ids]
        reps = [r for r in reps if r is not None]
        await asyncio.gather(*[r.abort_all_requests() for r in reps])
        self._record("abort", ids=[replica_id(r) for r in reps])
        return [replica_id(r) for r in reps]

    async def sleep_standalone(self, replica_ids: list[str]) -> list[str]:
        reps = [self._parked[rid] for rid in replica_ids if rid in self._parked]
        await asyncio.gather(*[r.sleep() for r in reps])
        self._record("sleep", ids=[replica_id(r) for r in reps])
        return [replica_id(r) for r in reps]

    async def destroy_standalone(self, replica_ids: list[str]) -> list[str]:
        """Kill the server + worker actors of PARKED replicas and remove their placement groups."""
        async with self._pool_lock:
            done = []
            for rid in replica_ids:
                r = self._parked.get(rid)
                if r is None:
                    continue
                for actor in list(r.servers) + list(r.workers):
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("kill %s failed: %s", actor, e)
                pool = getattr(r, "resource_pool", None)
                for pg in (getattr(pool, "pgs", None) or []):
                    try:
                        ray.util.remove_placement_group(pg)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("remove_placement_group failed: %s", e)
                self._parked.pop(rid, None)
                done.append(rid)
            self._record("destroy", ids=done)
            return done
