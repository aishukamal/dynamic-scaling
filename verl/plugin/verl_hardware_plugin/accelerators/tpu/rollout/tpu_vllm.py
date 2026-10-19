# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM rollout replica and HTTP server for Google TPU.

Upstream verl builds the vLLM rollout through ``RolloutReplicaRegistry``. On TPU the plugin
registers a ``vllm`` loader that returns :class:`TPUvLLMReplica`. It keeps upstream's HTTP server
and changes only what TPU needs:

* One server actor per replica, on the first worker's node (``launch_tpu_vllm_servers``). vLLM
  then spans the replica's hosts through its Ray executor, instead of verl's one server per node.
* Hybrid replicas (``init_hybrid``, the colocated mode): the trainer workers release their chips
  first, and vLLM's workers are placed on the trainer workers' nodes and chips by node affinity, as
  upstream's hybrid server inherits the trainer's ``CUDA_VISIBLE_DEVICES``. Disaggregated replicas
  keep the placement-group path.
* ``distributed_executor_backend=external_launcher``: the multi-host executor is selected by
  ``patch_vllm_for_tpu``. Sleep mode stays as upstream configures it; ``patch_vllm_for_tpu`` gives
  vllm-torchtpu's worker a memory-only ``sleep`` / ``wake_up`` while it has none (``tpu_sleep.py``),
  and the server picks sleep level 2 for the ``raiden`` / ``tpu`` checkpoint engines.
* With ``checkpoint_engine.backend=raiden``, vLLM's workers load the Raiden worker extension
  (``tpu_raiden.py``), which receives the trainer's weights over the network.

The same logic previously lived behind ``get_resource_name() == "TPU"`` branches in verl's
``vllm_async_server.py`` and ``replica.py``.
"""

import asyncio
import json
import logging
import os
from typing import Any, Callable, Optional

import ray

from verl.plugin.platform import get_platform
from verl.utils.device import get_device_name
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from verl_hardware_plugin.accelerators.tpu import platform_tpu
from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import patch_vllm_for_tpu

# vLLM imports the worker extension by name in its worker processes.
RAIDEN_WORKER_EXTENSION_CLS = "verl_hardware_plugin.accelerators.tpu.rollout.tpu_raiden.vLLMRaidenWorkerExtension"
# Hybrid replicas: "<node id>:<chip>" of every trainer worker of the replica, in rank order, read by
# _init_workers_ray in the EngineCore process. Disaggregated replicas pass placement groups instead.
HYBRID_WORKERS_ENV = "VERL_TPU_HYBRID_WORKERS"


def _trainer_worker_placement(worker_self: Any) -> tuple[str, str]:
    """Runs inside a trainer worker actor (``__ray_call__``): its Ray node id and the chip it was given."""
    return ray.get_runtime_context().get_node_id(), os.environ.get("TPU_VISIBLE_CHIPS", "0")


def _suspend_trainer_worker(worker_self: Any) -> dict:
    """Runs inside a trainer worker actor (``__ray_call__``): release its chip so vLLM can take it."""
    return platform_tpu.suspend_device()


# Multi-host hybrid replicas: a replica of TP > chips-per-host spans several trainer hosts, and on a multi-host
# slice those hosts must be one physical machine (sub-block) for the chips to form a mesh. The hosts are ordered
# by VERL_TPU_HOST_ORDER (comma-separated host IPs in physical order) when set, else by the VM's physical_host
# metadata attribute (the value the old phys.sh probe read); consecutive hosts in that order form a replica.
HOST_ORDER_ENV = "VERL_TPU_HOST_ORDER"
PHYSICAL_HOST_ENV = "VERL_TPU_PHYSICAL_HOST"
_METADATA_URL = "http://169.254.169.254/computeMetadata/v1/instance/attributes/physical_host"
_physical_host_cache: dict[str, Optional[str]] = {}

logger = logging.getLogger(__name__)


def _physical_host() -> Optional[str]:
    """This VM's ``physical_host`` metadata attribute (``/CCCC/BBBB/HHHH``: cluster, block, host), or None."""
    if PHYSICAL_HOST_ENV in os.environ:
        return os.environ[PHYSICAL_HOST_ENV] or None
    if "value" not in _physical_host_cache:
        import urllib.request

        try:
            req = urllib.request.Request(_METADATA_URL, headers={"Metadata-Flavor": "Google"})
            _physical_host_cache["value"] = urllib.request.urlopen(req, timeout=3).read().decode().strip() or None
        except Exception as exc:  # noqa: BLE001 - no metadata server (not GCE) or attribute absent
            logger.debug("physical_host metadata unavailable: %s", exc)
            _physical_host_cache["value"] = None
    return _physical_host_cache["value"]


def _trainer_worker_host(worker_self: Any) -> tuple[str, str, Optional[str]]:
    """Runs inside a trainer worker actor (``__ray_call__``): its Ray node id, node IP and physical host."""
    return ray.get_runtime_context().get_node_id(), ray.util.get_node_ip_address().strip("[]"), _physical_host()


def hybrid_replica_workers(workers: list, world_size: int, replica_rank: int, hosts: list[tuple]) -> tuple[list, str]:
    """The trainer workers that form hybrid replica ``replica_rank`` and how they were chosen.

    ``hosts[i] = (node_id, node_ip, physical_host)`` of ``workers[i]``. A replica that fits on one host takes
    upstream's slice of ``world_size`` consecutive workers (rank order). A replica that spans several hosts is
    formed from ``world_size // chips_per_host`` consecutive hosts in physical order: ``VERL_TPU_HOST_ORDER``
    (host IPs; hosts not listed sort after it, by IP) when set, else the hosts grouped by their ``physical_host``
    metadata (hosts of one machine are adjacent; machines in order of their first worker); without either, rank
    order with a warning. Workers within a replica keep their order (host, then chip).
    """
    by_node: dict[str, list[int]] = {}
    for idx, (node_id, _ip, _phys) in enumerate(hosts):
        by_node.setdefault(node_id, []).append(idx)
    node_ids = list(by_node)
    chips_per_host = len(by_node[node_ids[0]])
    if any(len(v) != chips_per_host for v in by_node.values()):
        raise ValueError(f"uneven trainer workers per host: { {n: len(v) for n, v in by_node.items()} }")
    if world_size <= chips_per_host:
        return workers[world_size * replica_rank : world_size * (replica_rank + 1)], "rank order"
    if world_size % chips_per_host:
        raise ValueError(f"replica world size {world_size} is not a multiple of the {chips_per_host} chips per host")
    hosts_per_replica = world_size // chips_per_host
    ip_of = {node_id: hosts[idxs[0]][1] for node_id, idxs in by_node.items()}
    phys_of = {node_id: hosts[idxs[0]][2] for node_id, idxs in by_node.items()}
    order = [ip.strip() for ip in os.environ.get(HOST_ORDER_ENV, "").split(",") if ip.strip()]
    if order:
        rank_of = {ip: i for i, ip in enumerate(order)}
        ordered = sorted(node_ids, key=lambda n: (0, rank_of[ip_of[n]], ip_of[n]) if ip_of[n] in rank_of else (1, 0, ip_of[n]))
        how = HOST_ORDER_ENV
    elif all(phys_of[n] for n in node_ids):
        first = {n: node_ids.index(n) for n in node_ids}
        machine_first = {}
        for n in node_ids:
            machine_first.setdefault(phys_of[n], first[n])
        ordered = sorted(node_ids, key=lambda n: (machine_first[phys_of[n]], phys_of[n], first[n]))
        how = "physical_host"
    else:
        ordered = node_ids
        how = "rank order"
        logger.warning(
            "Hybrid replica %d spans %d hosts but neither %s nor the physical_host metadata is available; "
            "using rank order, which may not be a physical machine pair",
            replica_rank, hosts_per_replica, HOST_ORDER_ENV,
        )
    chosen = ordered[hosts_per_replica * replica_rank : hosts_per_replica * (replica_rank + 1)]
    if len(chosen) != hosts_per_replica:
        raise ValueError(f"replica {replica_rank} needs {hosts_per_replica} hosts, {len(node_ids)} available")
    return [workers[i] for n in chosen for i in by_node[n]], how


async def launch_tpu_vllm_servers(replica: vLLMReplica) -> None:
    """Launch the vLLM rollout server actor for a TPU replica."""
    if replica.config.data_parallel_size > 1:
        raise NotImplementedError(
            "actor_rollout_ref.rollout.data_parallel_size="
            f"{replica.config.data_parallel_size} is not supported on TPU. The TPU "
            "distributed runtime requires each compiled program to span the whole "
            "TPU mesh, but a single DP group only spans "
            f"tensor_model_parallel_size={replica.config.tensor_model_parallel_size} "
            "chips. Set actor_rollout_ref.rollout.data_parallel_size=1 instead: verl "
            "then creates one rollout replica per tensor_model_parallel_size chips, "
            "which is equivalent to engine-internal data parallelism."
        )

    # Upstream starts one server per node and gives it the node's devices. On TPU one server, on the
    # first worker's node, drives all of the replica's hosts and holds no chips: patch_vllm_for_tpu
    # gives each of vLLM's own workers a chip.
    hybrid = replica.rollout_mode == RolloutMode.HYBRID
    if hybrid:
        # Hybrid replica: the workers are the trainer's actors. vLLM's workers go to the same nodes and
        # chips, exactly as upstream's hybrid server inherits CUDA_VISIBLE_DEVICES from the trainer
        # workers (vllm_async_server.py launch_servers); the trainer actor keeps the bundle's TPU:1.
        placements = await asyncio.gather(*[w.__ray_call__.remote(_trainer_worker_placement) for w in replica.workers])
        node_id = placements[0][0]
    else:
        node_id = await replica.workers[0].__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())

    prefix = replica._get_server_name_prefix()
    if replica.is_reward_model:
        name = f"{prefix}server_reward_{replica.replica_rank}_0{replica.name_suffix}"
    elif replica.is_teacher_model:
        name = f"{prefix}server_teacher_{replica.replica_rank}_0{replica.name_suffix}"
    else:
        name = f"{prefix}server_{replica.replica_rank}_0{replica.name_suffix}"

    env_vars = {
        **{var: "1" for var in get_platform().ray_noset_envvars()},
        **get_platform().rollout_env_vars(),
        # One engine spans all of the replica's hosts through Ray. patch_vllm_for_tpu builds on
        # vllm-torchtpu's Ray multi-host backend with the V1 executor, not the V2 one.
        "TPU_MULTIHOST_BACKEND": "ray",
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "0",
    }
    if hybrid:
        env_vars[HYBRID_WORKERS_ENV] = ",".join(f"{node}:{chip}" for node, chip in placements)
    else:
        # The initialize_ray_cluster patch in patch_vllm_for_tpu attaches vLLM to these placement groups.
        pgs = replica.resource_pool.get_placement_groups(device_name=get_device_name())
        env_vars["VERL_TPU_PG_IDS"] = ",".join(pg.id.hex() for pg in pgs)
        # Several single-host replicas on one host (optionally next to a trainer pool on the same host): each
        # replica drives chips ``offset + replica_rank * TP + tp_rank``; without the knob a worker drives chip
        # ``local rank within its replica``, which collides as soon as two replicas share a host.
        chip_offset = os.environ.get("VERL_TPU_ROLLOUT_CHIP_OFFSET")
        if chip_offset:
            env_vars["VERL_TPU_ROLLOUT_CHIP_BASE"] = str(int(chip_offset) + replica.replica_rank * replica.world_size)
        # Per-replica worker environment (pool-control experiment: e.g. a fresh TORCHINDUCTOR_CACHE_DIR for a cold
        # spawn). The server/EngineCore get it here; _init_workers_ray forwards it to the replica's vLLM workers.
        pool_env = os.environ.get(f"VERL_TPU_POOL_WORKER_ENV_{replica.replica_rank}")
        if pool_env:
            env_vars["VERL_TPU_POOL_WORKER_ENV"] = pool_env
            env_vars.update(json.loads(pool_env))

    server = replica.server_class.options(
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False,
        ),
        runtime_env={"env_vars": env_vars},
        name=name,
        max_concurrency=replica.config.ray_actor_max_concurrency,
    ).remote(
        config=replica.config,
        model_config=replica.model_config,
        rollout_mode=replica.rollout_mode,
        workers=replica.workers,
        replica_rank=replica.replica_rank,
        node_rank=0,
        gpus_per_node=replica.gpus_per_replica_node,
        nnodes=replica.nnodes,
        cuda_visible_devices="",
    )
    replica.servers.append(server)

    master_address, master_port, dp_rpc_port = await replica.servers[0].get_master_address.remote()
    await replica.servers[0].launch_server.remote(
        master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
    )

    server_address, server_port = await replica.servers[0].get_server_address.remote()
    replica._server_handle = replica.servers[0]
    replica._server_address = (
        f"[{server_address}]:{server_port}"
        if is_valid_ipv6_address(server_address)
        else f"{server_address}:{server_port}"
    )


class TPUvLLMHttpServer(vLLMHttpServer):
    """``vLLMHttpServer`` with the TPU engine arguments."""

    def __init__(self, *args, **kwargs):
        # This actor builds the vLLM engine and starts its EngineCore process, so patch vLLM before
        # anything else. The patches carry themselves into the EngineCore and vLLM's workers.
        patch_vllm_for_tpu()
        super().__init__(*args, **kwargs)

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        super()._preprocess_engine_kwargs(engine_kwargs)
        # engine_kwargs is merged last into the vLLM CLI args, so these win over the defaults.
        # patch_vllm_for_tpu switches the engine to vLLM's Ray executor at config time.
        engine_kwargs["distributed_executor_backend"] = "external_launcher"

    def _resolve_sleep_level(self) -> int:
        """Sleep level for the hybrid sleep after sampling.

        With the ``raiden`` or ``tpu`` checkpoint engine every parameter is rewritten by the install that
        follows the wake, so the weights need no host copy: level 2 (KV cache and weights freed). Measured
        on Qwen3-14B, TP=4: the sampler's suspended image shrinks from about 24 GiB to under 0.2 GiB per
        worker when both are dropped before the device is released. Other backends keep upstream's choice.
        """
        checkpoint_engine = getattr(getattr(self, "config", None), "checkpoint_engine", None)
        if getattr(checkpoint_engine, "backend", None) in ("raiden", "tpu"):
            return 2
        return super()._resolve_sleep_level()

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        # Upstream's collective_rpc discards the workers' return values (verl PR #8155 returns them).
        # The TPU weight sync reads them: per-worker install timings, parity stats and the hand-over
        # methods' timing dicts.
        return await self.engine.collective_rpc(method=method, timeout=timeout, args=args, kwargs=kwargs)

    def _get_worker_extension_cls(self) -> str:
        # The raiden checkpoint engine transfers weights straight into vLLM's workers, which need the
        # methods of the Raiden worker extension. Other backends keep upstream's extension.
        checkpoint_engine = getattr(getattr(self, "config", None), "checkpoint_engine", None)
        if getattr(checkpoint_engine, "backend", None) == "raiden":
            return RAIDEN_WORKER_EXTENSION_CLS
        return super()._get_worker_extension_cls()


class TPUvLLMReplica(vLLMReplica):
    """vLLM rollout replica for TPU: one server actor drives all of the replica's hosts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(TPUvLLMHttpServer)

    async def init_hybrid(self, worker_group):
        """Hybrid replica on the trainer's own chips.

        A TPU chip has one owner process, so the trainer workers of this replica release their chips
        (``platform_tpu.suspend_device`` through ``__ray_call__``, the facility upstream uses to read the
        workers' node ids) before the vLLM engine initializes and compiles on them. The call is
        idempotent; the colocated checkpoint engine manager resumes the trainer after the first
        ``sleep_replicas``. Everything else is upstream's ``init_hybrid``.
        """
        from verl_hardware_plugin.accelerators.tpu.engines.tpu_colocated import install_manager_selector

        # Until verl main resolves rollout.checkpoint_manager_class (C1), bind the selector that picks the
        # colocated manager into the V1 trainer module, which is imported by the time hybrid replicas exist.
        install_manager_selector()
        hosts = await asyncio.gather(*[w.__ray_call__.remote(_trainer_worker_host) for w in worker_group.workers])
        workers, how = hybrid_replica_workers(worker_group.workers, self.world_size, self.replica_rank, list(hosts))
        await asyncio.gather(*[worker.__ray_call__.remote(_suspend_trainer_worker) for worker in workers])
        if how == "rank order":
            await super().init_hybrid(worker_group)
            return
        # Multi-host replica on physically paired hosts: upstream's init_hybrid would slice by rank order.
        logger.info(
            "Hybrid replica %d: %d trainer workers on hosts %s (ordered by %s)",
            self.replica_rank, len(workers), sorted({hosts[worker_group.workers.index(w)][1] for w in workers}), how,
        )
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = workers
        await self.launch_servers()

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )
        await launch_tpu_vllm_servers(self)
