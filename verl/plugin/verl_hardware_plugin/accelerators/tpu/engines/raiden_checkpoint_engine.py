# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Raiden (tpu-sync) checkpoint engine: peer-to-peer weight sync between TPU slices.

The ``tpu`` checkpoint engine stages the whole model through Ray's object store. ``raiden`` instead
moves the weights over the network from the trainer chips to the rollout chips with the Raiden transfer
library (``tpu-sync-torch``, https://github.com/google/tpu-sync). Every trainer rank registers the full
weights that verl gathers for it, and each rollout chip receives only its vLLM tensor-parallel shard::

     [Trainer slice: FSDP ranks]                       [Rollout slice: vLLM TP workers]
    +-------------------------------------+           +--------------------------------------+
    | RaidenCheckpointEngine.send_weights |           | vLLMRaidenWorkerExtension            |
    |  - bind device tensors to a         |           |  - init_raiden_sync_on_worker:       |
    |    WeightSynchronizer               |           |    allocate TP-sharded receive       |
    |  - register "trainer/<rank>"        |           |    buffers, register "sampler/<rank>"|
    +------------------+------------------+           |  - install_raiden_weights: H2D and   |
                       |                              |    fuse into vLLM's parameters       |
                       +-- host-to-host, resharded -->+--------------------------------------+
                                     ^
                     [Driver: update_raiden_weights + RaidenController]

``update_raiden_weights`` runs on the driver in place of ``CheckpointEngineManager.update_weights``
(see ``apply_tpu_checkpoint_engine_hooks``). Trainer and rollout processes find the controller, and
the tensor shapes, through the ``RayWeightRegistry`` actor.

Security: the controller and the WeightSynchronizer endpoints accept unauthenticated connections and
carry model weights in the clear. Run them only on a trusted cluster network.
"""

import asyncio
import gc
import logging
import os
import time
from collections.abc import Iterable
from typing import Any, Optional

import ray
import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl_hardware_plugin.accelerators.tpu.engines.ray_weight_registry import get_ray_weight_registry

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

RAIDEN_BACKEND = "raiden"
# Parallel transfer streams per worker, unless engine_kwargs.raiden.parallelism overrides it.
RAIDEN_DEFAULT_PARALLELISM = 8
# How long the driver waits for every trainer and rollout worker to register with the controller.
RAIDEN_REGISTRATION_TIMEOUT_S = 60.0
# Raiden reads and writes device buffers in place, so the tensors it is given must live on this device.
RAIDEN_DEVICE = "tpu"


def tpu_synchronize(strict: bool = False) -> None:
    """Waits for pending TPU work, so Raiden reads final device buffers.

    Args:
        strict: Raise if ``torch_tpu`` cannot synchronize instead of logging a warning.
    """
    try:
        from torch_tpu._internal import sync as torch_tpu_sync

        torch_tpu_sync.synchronize(wait=True)
    except Exception as e:
        if strict:
            raise RuntimeError(f"TPU synchronization failed: {e}") from e
        logger.warning(f"Could not synchronize via torch_tpu: {e}")


def compute_tensor_stats(items: list) -> dict:
    """Compute deterministic L1/L2 norms and parameter counts across tensors.

    Reductions run on the device holding the tensors, to avoid copying the weights to the host.
    ``items`` holds ``(name, tensor)`` pairs (named tensors get a ``per_tensor`` entry) or bare tensors.
    """
    total_numel = 0
    total_l1 = 0.0
    total_l2_sq = 0.0
    per_tensor = {}

    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            name, p = item
        else:
            name = None
            p = item
        p_local = p.to_local() if hasattr(p, "to_local") else p

        # Reduce in float32 for precision.
        p_float = p_local.float()
        t_l1 = float(p_float.abs().sum().item())
        t_l2_sq = float(p_float.pow(2).sum().item())
        numel = p_local.numel()

        if name is not None:
            per_tensor[name] = {
                "l1": t_l1,
                "l2": float(t_l2_sq**0.5),
                "l2_sq": t_l2_sq,
                "numel": numel,
                "shape": list(p_local.shape),
                "dtype": str(p_local.dtype),
            }
        total_numel += numel
        total_l1 += t_l1
        total_l2_sq += t_l2_sq

    return {
        "total_numel": total_numel,
        "num_tensors": len(items),
        "l1_norm": total_l1,
        "l2_norm": float(total_l2_sq**0.5),
        "l2_sq": total_l2_sq,
        "per_tensor": per_tensor,
    }


def create_torch_weight_synchronizer(
    device_tensors: list[list[torch.Tensor]],
    local_port: int = 0,
    parallelism: int = RAIDEN_DEFAULT_PARALLELISM,
    listener_port: int = 0,
    bind_ip: str = "127.0.0.1",
) -> Any:
    """Creates a tpu_sync ``WeightSynchronizer`` over ``device_tensors``."""
    from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer

    return WeightSynchronizer(
        device_tensors,
        local_port=local_port,
        parallelism=parallelism,
        listener_port=listener_port,
        bind_ip=bind_ip,
        unsafe_skip_buffer_lock=True,
        auto_h2d=False,
    )


def _unwrap_tensor(t: Any) -> Any:
    """Extract the raw underlying local tensor from nn.Parameter or DTensor wrappers."""
    t = t.to_local().data if hasattr(t, "to_local") else (t.data if hasattr(t, "data") else t)
    return t


def filter_tied_embeddings(
    named_items: Iterable[tuple[str, Any]], tie_word_embeddings: bool = True
) -> list[tuple[str, Any]]:
    """Drop ``lm_head.weight`` from the weights to send when it is tied to the input embedding.

    For tied models (``tie_word_embeddings=True``, e.g. Qwen3-0.6B / 4B), ``lm_head.weight`` is the same
    tensor as ``embed_tokens.weight``, so it is not sent; the sampler copies ``embed_tokens`` into its
    ``lm_head`` after receiving the weights.

    For untied models (``tie_word_embeddings=False``, e.g. Qwen3-8B and larger), ``lm_head.weight`` is a
    separate trained weight and is kept. Dropping it would leave the sampler with a wrong ``lm_head``
    (stale, or overwritten with ``embed_tokens``).

    Args:
        named_items: ``(name, tensor)`` pairs in HF naming.
        tie_word_embeddings: ``hf_config.tie_word_embeddings`` of the trained model.

    Returns:
        The ``(name, tensor)`` pairs to send.
    """
    items = list(named_items)
    has_embed = any("embed_tokens" in k or "tok_embeddings" in k for k, _ in items)
    if has_embed and tie_word_embeddings:
        items = [(k, v) for k, v in items if not (k == "lm_head.weight" or k.endswith(".lm_head.weight"))]
    return items


def validate_and_sanitize_tensors(
    named_tensors: Iterable[tuple[str, Any]],
    device: Optional[torch.device] = None,
) -> list[tuple[str, torch.Tensor]]:
    """Validate and sanitize tensors for zero-copy DMA registration with Raiden.

    Performs physical memory validation:
    1. Drops None, non-tensor objects, zero-element tensors, and unallocated meta tensors.
    2. Unwraps DTensor/nn.Parameter to local tensor buffers.
    3. Ensures tensors physically reside on ``device`` (TPU HBM by default).
    4. Guarantees memory contiguity (contiguous buffers) required for direct DMA.
    """
    if device is None:
        device = torch.device(RAIDEN_DEVICE)

    sanitized = []
    for item in named_tensors:
        name, p = item[0], item[1]
        if p is None:
            continue
        t = _unwrap_tensor(p)
        if not isinstance(t, torch.Tensor):
            continue

        # Skip 0-element tensors and unallocated meta tensors (prevents C++ nullptr faults)
        if t.numel() == 0 or getattr(t, "is_meta", False):
            continue

        # Ensure the tensor physically resides in device memory
        if t.device.type != device.type:
            try:
                t = t.to(device)
            except Exception as e:
                logger.warning(f"Could not move {name} to {device}: {e}")
                continue

        # Ensure contiguous memory layout (required for direct DMA pointer calculation)
        if not t.is_contiguous():
            t = t.contiguous()

        sanitized.append((name, t))

    return sanitized


def setup_raiden_controller() -> tuple[Any, Any, str]:
    """Start a Raiden controller server in this process and publish its address in ``RayWeightRegistry``.

    Returns:
        ``(controller, server, "host:port")``. The caller must keep ``server`` referenced while syncing.
    """
    from tpu_sync.rpc import raiden_controller

    # TODO(security): the controller (like the WeightSynchronizer endpoints) accepts unauthenticated
    # connections. Add authentication once tpu_sync supports it; until then it must only be reachable
    # from the cluster's trusted network.
    controller = raiden_controller.RaidenController(port=0)
    server = raiden_controller.RaidenControllerServer(controller)
    port = server.start()
    ip = ray.util.get_node_ip_address().strip("[]")
    address = f"{ip}:{port}"
    logger.info(f"RaidenControllerServer started on the driver: {address}")

    try:
        registry = get_ray_weight_registry()
        ray.get(registry.set_controller_address.remote(address))
        logger.info(f"Stored the RaidenController address ({address}) in RayWeightRegistry")
    except Exception as reg_err:
        raise RuntimeError(
            f"Failed to store RaidenController address ({address}) in RayWeightRegistry: {reg_err}"
        ) from reg_err
    return controller, server, address


def raiden_is_tile_aligned(local_shape: list) -> bool:
    """True when a 2D+ shard matches the TPU (8, 128) tile, so it can skip the CPU (de)tiling pass."""
    return (len(local_shape) >= 2) and (local_shape[-1] % 128 == 0) and (local_shape[-2] % 8 == 0)


def apply_raiden_skip_tiling(ws: Any, skip_tiling_plan: list) -> None:
    """Set the per-tensor skip_tiling plan on a WeightSynchronizer (API name differs across tpu_sync builds)."""
    if hasattr(ws, "test_only_set_skip_tiling"):
        ws.test_only_set_skip_tiling(skip_tiling_plan)
    elif hasattr(ws, "set_skip_tiling"):
        ws.set_skip_tiling(skip_tiling_plan)


def _as_bool(value: Any) -> bool:
    """Parse a bool flag that may arrive as a string (env var, CLI override) as well as a bool."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _dim0_shard_info(spec: Any) -> Optional[tuple[tuple, int, int]]:
    """Return ``(local_shape, num_shards, shard_index)`` if the local shard is an even dim-0 cut on a 1-D mesh.

    That is the FSDP2 ``Shard(0)`` layout, which Raiden can describe directly: the variable keeps its full
    shape, is split ``num_shards`` ways on dim 0, and this rank holds block ``shard_index``. Uneven cuts
    are excluded because torch gives the remainder to the leading shards while Raiden gives it to the last.
    """
    from verl.workers.engine.spec import BlockPlacement, derive_dtensor_placement

    if spec.mesh is None or spec.mesh.ndim != 1:
        return None
    place, _, _ = derive_dtensor_placement(spec)
    if not isinstance(place, BlockPlacement) or not place.is_flat_contiguous:
        return None
    num_shards = spec.mesh.size(0)
    full_dim0 = int(spec.full_shape[0])
    if num_shards <= 1 or full_dim0 % num_shards:
        return None
    return tuple(place.local_shape), num_shards, int(place.global_offset[0]) // (full_dim0 // num_shards)


def export_local_shards(
    engine: Any, tie_word_embeddings: bool = True
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, tuple[int, int]]]:
    """Export this rank's weights for Raiden without all-gathering FSDP shards.

    Returns ``(named_tensors, shard_info)``. ``shard_info[name] = (num_shards, shard_index)`` for tensors
    exported as a local dim-0 shard; tensors absent from it are full (replicated) on every rank. Tensors
    whose layout Raiden cannot express are all-gathered here, in the same order on every rank.
    ``lm_head.weight`` is dropped only for tied models (see ``filter_tied_embeddings``).
    """
    from torch.distributed.tensor import DTensor

    from verl.workers.engine.spec import BlockPlacement, derive_dtensor_placement

    gen, _ = engine.get_per_tensor_param_shard()
    items = filter_tied_embeddings(
        ((name, (local, spec)) for name, local, spec in gen), tie_word_embeddings=tie_word_embeddings
    )

    named, shard_info = [], {}
    for name, (local, spec) in items:
        if spec.place is not None or spec.hf_slots is not None:
            raise NotImplementedError(
                f"Raiden sharded export does not support exporter-defined placements or expert stacks ({name})"
            )
        full_shape = tuple(int(d) for d in spec.full_shape)
        info = _dim0_shard_info(spec)
        if info is not None:
            local_shape, num_shards, shard_index = info
            named.append((name, local.view(local_shape)))
            shard_info[name] = (num_shards, shard_index)
            continue
        place = derive_dtensor_placement(spec)[0] if spec.mesh is not None else 0
        if not isinstance(place, BlockPlacement):
            # unsharded or fully replicated: the local tensor is already the full parameter
            named.append((name, local.view(full_shape)))
            continue
        strides = [1] * len(full_shape)
        for d in range(len(full_shape) - 2, -1, -1):
            strides[d] = strides[d + 1] * full_shape[d + 1]
        dt = DTensor.from_local(
            local.view(place.local_shape),
            spec.mesh,
            spec.placements,
            run_check=False,
            shape=torch.Size(full_shape),
            stride=tuple(strides),
        )
        named.append((name, dt.full_tensor()))
    return named, shard_info


def merge_rank_stats(rank_stats: dict) -> dict:
    """Sum per-tensor partial stats (l1, l2_sq, numel) posted by every trainer rank."""
    per_tensor: dict[str, dict[str, float]] = {}
    for stats in rank_stats.values():
        for name, t in stats.get("per_tensor", {}).items():
            acc = per_tensor.setdefault(name, {"l1": 0.0, "l2_sq": 0.0, "numel": 0})
            acc["l1"] += t["l1"]
            acc["l2_sq"] += t.get("l2_sq", t.get("l2", 0.0) ** 2)
            acc["numel"] += t["numel"]
    for acc in per_tensor.values():
        acc["l2"] = acc["l2_sq"] ** 0.5
    total_l2_sq = sum(t["l2_sq"] for t in per_tensor.values())
    return {
        "total_numel": sum(t["numel"] for t in per_tensor.values()),
        "num_tensors": len(per_tensor),
        "l1_norm": sum(t["l1"] for t in per_tensor.values()),
        "l2_norm": total_l2_sq**0.5,
        "l2_sq": total_l2_sq,
        "per_tensor": per_tensor,
    }


@CheckpointEngineRegistry.register(RAIDEN_BACKEND)
class RaidenCheckpointEngine(CheckpointEngine):
    """Trainer-side Raiden engine: exposes this rank's weights to the controller-driven P2P transfer.

    Engine kwargs (``actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden``):

    * ``parallelism`` (default 8): parallel transfer streams per worker.
    * ``verify_parity`` (default False): post weight norms after every sync so the driver can check
      that the rollout received the trainer's weights.
    * ``tie_word_embeddings`` (default unset): whether ``lm_head`` is tied to the input embedding, so it
      need not be sent. Unset sends ``lm_head``, which is correct for tied and untied models alike.
    * ``release_buffers_after_sync`` (default True): free the bound send tensors after every transfer
      (``release_sync_buffers``) instead of keeping them resident until the next sync. The environment
      variable ``VERL_RAIDEN_RELEASE_BUFFERS`` overrides it.
    """

    # The worker hands send_weights the training engine (verl engine_workers.update_weights), so each rank
    # exports and registers its local FSDP shard instead of an all-gathered copy of the whole model.
    consumes_training_engine = True

    def __init__(self, bucket_size: int = 0, is_master: bool = False, **kwargs: Any) -> None:
        # CheckpointEngine defines no __init__ (object.__init__ takes no arguments).
        self.bucket_size = bucket_size
        self.is_master = is_master
        self.parallelism = int(kwargs.get("parallelism", RAIDEN_DEFAULT_PARALLELISM))
        self.verify_parity = _as_bool(kwargs.get("verify_parity", False))
        tie_word_embeddings = kwargs.get("tie_word_embeddings")
        self.tie_word_embeddings = None if tie_word_embeddings is None else _as_bool(tie_word_embeddings)
        # The synchronizer pins the device buffers of every bound tensor (a bf16 copy of the full model on
        # every trainer chip), so dropping the Python references alone frees nothing. On by default so the
        # trainer keeps that HBM headroom between syncs; opt out with the engine kwarg or
        # VERL_RAIDEN_RELEASE_BUFFERS=0 to keep the send tensors resident.
        release = _as_bool(kwargs.get("release_buffers_after_sync", True))
        env_release = os.environ.get("VERL_RAIDEN_RELEASE_BUFFERS", "")
        if env_release:
            release = _as_bool(env_release)
        self.release_buffers_after_sync = release
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = int(os.environ.get("RANK", "0"))

        self._registry: Any = None
        self._controller_addr: Optional[str] = None
        self._trainer_raiden_ws: Any = None
        # (name, shape, dtype, shard) of the tensors the live synchronizer was created and registered with; a
        # later send_weights with the same signature rebinds instead of re-creating it.
        self._registered_signature: Optional[list[tuple]] = None
        # Keeps the bound device buffers alive until release_sync_buffers (the controller-driven push reads
        # them after send_weights returns).
        self._bound_tensors: Optional[list[torch.Tensor]] = None
        # name -> (num_shards, shard_index) for the tensors of the last send registered as local dim-0 shards.
        self._shard_info: dict[str, tuple[int, int]] = {}
        self._warned_no_unbind = False

    @property
    def registry(self) -> Any:
        """The ``RayWeightRegistry`` actor, looked up on first use."""
        if self._registry is None:
            self._registry = get_ray_weight_registry()
        return self._registry

    def prepare(self) -> dict:
        return {}

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        return {}, {}

    def init_process_group(self, **kwargs: Any) -> None:
        pass

    def finalize(self) -> None:
        self._close_synchronizer()

    def _close_synchronizer(self) -> None:
        ws, self._trainer_raiden_ws = self._trainer_raiden_ws, None
        self._registered_signature = None
        self._bound_tensors = None
        close = getattr(ws, "close", None)
        if close is not None:
            try:
                close()
            except Exception as e:
                logger.debug(f"Trainer Rank {self.rank}: closing the WeightSynchronizer failed: {e}")

    async def send_weights(self, weights: Any, global_steps: Optional[int] = None) -> None:
        """Bind this rank's weights to its WeightSynchronizer, register them with the controller and stage them
        in host memory.

        Args:
            weights: The training engine (``consumes_training_engine``): every rank exports its local FSDP shards
                through ``get_per_tensor_param_shard`` and registers them with their full shape, mesh_shape
                ``[N, 1, ...]`` and ``global_shard_indices=[i]``; Raiden reshards FSDP -> sampler TP during the
                push. An iterable (or dict) of ``(hf_name, tensor)`` pairs is accepted too: those are the full
                weights, registered unsharded on every rank.
            global_steps: Trainer step of these weights.

        The transfer itself is driven by the controller after this returns, so the bound tensors are kept until
        ``release_sync_buffers``. The synchronizer is created and registered on the first sync and reused (rebound
        through ``bind_weights``) afterwards as long as the tensor names, shapes and dtypes do not change.
        """
        step_key = global_steps if global_steps is not None else 0
        logger.info(f"RaidenCheckpointEngine: [Step {step_key}] Start send_weights...")
        tpu_synchronize(strict=True)

        if hasattr(weights, "get_per_tensor_param_shard"):
            # Handed the training engine: register each rank's local FSDP shard, no all-gather.
            tie = self.tie_word_embeddings
            if tie is None:
                hf_config = getattr(getattr(weights, "model_config", None), "hf_config", None)
                tie = bool(getattr(hf_config, "tie_word_embeddings", False))
            named_weights, self._shard_info = export_local_shards(weights, tie_word_embeddings=bool(tie))
        else:
            # Materialize weights in a single pass and filter tied embeddings
            named_weights = filter_tied_embeddings(
                weights.items() if hasattr(weights, "items") else weights,
                tie_word_embeddings=bool(self.tie_word_embeddings),
            )
            self._shard_info = {}

        # Trainer sends pure canonical un-fused model weights directly
        unfused_weights = {k: _unwrap_tensor(v) for k, v in named_weights}

        sorted_weights = sorted(unfused_weights.items(), key=lambda x: x[0])
        valid_weights = validate_and_sanitize_tensors(sorted_weights, device=torch.device(RAIDEN_DEVICE))

        tpu_synchronize(strict=True)

        # Reuse the synchronizer when the tensor signature is unchanged: rebinding keeps the pinned host staging
        # buffers, listener threads and controller registration, and only swaps the device buffers the D2H reads
        # from (release_sync_buffers unbinds them between syncs). Re-creating it every sync re-allocates, pins
        # and first-touches a model-sized host buffer per rank and re-registers with the controller.
        signature = [(name, tuple(t.shape), t.dtype, self._shard_info.get(name)) for name, t in valid_weights]
        if self._trainer_raiden_ws is not None and signature == self._registered_signature:
            logger.info(f"Trainer Rank {self.rank}: rebinding {len(valid_weights)} tensors to the WeightSynchronizer")
            self._trainer_raiden_ws.bind_weights([[t] for _, t in valid_weights])
        else:
            await self._create_and_register(valid_weights)
            self._registered_signature = signature
        self._bound_tensors = [t for _, t in valid_weights]

        # No explicit ws.d2h() here: PushWeightsResharded (triggered by the controller's start_transfer)
        # performs a pipelined D2H overlapped with H2H, using the per-transfer skip_tiling plan. An
        # explicit d2h() before the first transfer has no skip_tiling plan yet and stages the wrong
        # (detiled) layout; it was only ever masked by the push's own D2H.

        if not self.verify_parity:
            return
        try:
            if self._shard_info:
                # Every rank posts partial stats: its local shards, plus the full tensors on rank 0 only.
                # The orchestrator sums them per tensor once every rank has posted.
                items = [(n, t) for n, t in valid_weights if n in self._shard_info or self.rank == 0]
                trainer_stats = compute_tensor_stats(items)
                trainer_stats["rank"] = self.rank
                self.registry.set_rank_stats.remote(step_key, self.rank, trainer_stats)
            elif self.is_master:
                # Every rank holds the full model: rank 0's stats are the reference.
                trainer_stats = compute_tensor_stats(valid_weights)
                trainer_stats["rank"] = self.rank
                self.registry.set_stats.remote(step_key, trainer_stats)
            else:
                return
            logger.info(
                f"[RAIDEN PARITY] Successfully stored trainer rank {self.rank} stats for step {step_key}: "
                f"L1={trainer_stats['l1_norm']:.4f}, numel={trainer_stats['total_numel']}"
            )
        except Exception as e:
            logger.warning(f"Failed to record trainer rank {self.rank} stats in RayWeightRegistry: {e}")

    async def _create_and_register(self, valid_weights: list[tuple[str, torch.Tensor]]) -> None:
        """(Re)creates the WeightSynchronizer over ``valid_weights`` and registers it with the controller."""
        self._close_synchronizer()

        logger.info(f"Trainer Rank {self.rank}: binding {len(valid_weights)} tensors to WeightSynchronizer")
        bind_ip = ray.util.get_node_ip_address().strip("[]")
        self._trainer_raiden_ws = create_torch_weight_synchronizer(
            [[t] for _, t in valid_weights],
            local_port=0,
            listener_port=0,
            parallelism=self.parallelism,
            bind_ip=bind_ip,
        )

        def _full_shape(name: str, p: torch.Tensor) -> list[int]:
            shape = list(p.shape)
            if name in self._shard_info:
                shape[0] *= self._shard_info[name][0]
            return shape

        # Record global (full) shapes in RayWeightRegistry so the sampler can size its receive buffers.
        # TODO(tpu): Move global_shapes registration to a one-time setup step during initialization
        # (e.g., in prepare or build_process_group/model_init) instead of per weight-sync step,
        # since tensor shapes are static across training iterations and only need to be communicated once.
        try:
            global_shapes = {name: _full_shape(name, p) for name, p in valid_weights}
            await self.registry.set_global_shapes.remote(global_shapes)
        except Exception as e:
            logger.warning(f"Could not record global shapes in RayWeightRegistry: {e}")

        # Build variable metadata protos for each dynamic tensor. Every variable carries its full shape.
        # - Local dim-0 shard: mesh_shape=[N, 1, ...] and global_shard_indices=[i] (this rank holds block i
        #   of N); Raiden reshards it to the sampler layout.
        # - Full tensor (replicated on every rank): mesh_shape=[1] * rank, unsharded.
        from tpu_sync.rpc import raiden_controller, raiden_service_pb2

        variable_protos = []
        max_shards = 1
        for idx, (name, p) in enumerate(valid_weights):
            shape = _full_shape(name, p)
            if name in self._shard_info:
                num_shards, shard_index = self._shard_info[name]
                max_shards = max(max_shards, num_shards)
                extra = {
                    "mesh_shape": [num_shards] + [1] * (len(shape) - 1),
                    "sharding_spec": ["fsdp"] + [""] * (len(shape) - 1),
                    "global_shard_indices": [shard_index],
                }
            else:
                extra = {"mesh_shape": [1] * len(shape), "sharding_spec": [""] * len(shape)}
            variable_protos.append(
                raiden_service_pb2.VariableMetadataProto(
                    name=name,
                    shape=shape,
                    layout=list(range(len(shape) - 1, -1, -1)),
                    item_size=p.element_size(),
                    layer_idx=idx,
                    **extra,
                )
            )
        num_sharded = sum(name in self._shard_info for name, _ in valid_weights)
        logger.info(
            f"Trainer Rank {self.rank}: registering {num_sharded} sharded and "
            f"{len(valid_weights) - num_sharded} full tensors"
        )

        # TODO(tpu): Consider passing controller_address directly during orchestration (e.g., via
        # CheckpointEngineManager / actor_wg.update_weights or init_process_group) rather than querying
        # the RayWeightRegistry actor, eliminating cross-process registry lookups altogether.
        if self._controller_addr is None:
            for _ in range(30):
                try:
                    self._controller_addr = await self.registry.get_controller_address.remote()
                    if self._controller_addr:
                        break
                except Exception as e:
                    logger.debug(f"Trainer Rank {self.rank}: controller address lookup failed: {e}")
                await asyncio.sleep(0.1)
        if not self._controller_addr:
            raise RuntimeError(
                f"Trainer Rank {self.rank}: No RaidenController address found in RayWeightRegistry after timeout"
            )

        try:
            ctrl_client = raiden_controller.RaidenControllerClientFacade(self._controller_addr)
            unit_id = raiden_controller.RaidenId("trainer", str(self.rank), "weights")
            ctrl_client.register_work_unit(
                unit_id,
                [f"{bind_ip}:{self._trainer_raiden_ws.local_port}"],
                f"{bind_ip}:{self._trainer_raiden_ws.listener_port}",
                mesh_shape=[max_shards, 1],
                variables=variable_protos,
                mesh_axes=["fsdp", "tp"],
            )
        except Exception as reg_err:
            logger.error(
                f"Trainer Rank {self.rank} failed to register with RaidenController "
                f"({self._controller_addr}): {reg_err}"
            )
            raise
        logger.info(
            f"Trainer Rank {self.rank} bound {len(valid_weights)} dynamic tensors and registered "
            f"directly with RaidenController ({self._controller_addr}): "
            f"data_port={self._trainer_raiden_ws.local_port}, "
            f"listener_port={self._trainer_raiden_ws.listener_port}"
        )

    def release_sync_buffers(self) -> dict:
        """Free this rank's bound send tensors after the transfer, keeping the WeightSynchronizer when possible.

        Must only be called after the controller transfer that reads them has completed. A no-op when
        ``release_buffers_after_sync`` is disabled, in which case the bound tensors stay resident between syncs.

        With a ``tpu_sync`` build that has ``WeightSynchronizer.unbind_weights()`` the synchronizer's holds on the
        bound device buffers are dropped, so dropping our own references returns the HBM. The synchronizer itself,
        its pinned host staging buffers and its controller registration survive, and the next ``send_weights``
        rebinds through ``bind_weights()``. Compared to destroying and re-creating the synchronizer each sync, this
        removes the per-step rebuild (host buffer allocation, pinning and first touch, plus controller
        re-registration: ~2 s "Trainer init" + ~2 s release on Qwen3-8B, v6e-8) from the sync critical path.

        Older ``tpu_sync`` builds without ``unbind_weights`` fall back to destroying the synchronizer; the next
        ``send_weights`` then re-creates and re-registers it.
        """
        if not self.release_buffers_after_sync or self._trainer_raiden_ws is None:
            return {}
        if hasattr(self._trainer_raiden_ws, "unbind_weights"):
            self._trainer_raiden_ws.unbind_weights()
            self._bound_tensors = None
        else:
            if not self._warned_no_unbind:
                logger.warning(
                    f"Trainer Rank {self.rank}: tpu_sync WeightSynchronizer has no unbind_weights(); destroying "
                    "it to free the send buffers each sync (upgrade tpu-sync-torch to keep it across syncs)."
                )
                self._warned_no_unbind = True
            # The torch WeightSynchronizer has no close(); the C++ object (which holds references to the
            # device tensors and the host staging memory) is destroyed when the last Python reference goes.
            self._close_synchronizer()
            gc.collect()
        # No empty_cache(): on TPU it clears the eager-op compilation cache (forcing recompiles) and the
        # TPU runtime reuses freed HBM without it.
        tpu_synchronize()
        return {}

    def receive_weights(self, global_steps: Optional[int] = None, **kwargs: Any):
        raise NotImplementedError(
            "Rollout workers receive Raiden weights through vLLMRaidenWorkerExtension via collective_rpc."
        )


def _raiden_engine_kwargs(config: Any) -> dict[str, Any]:
    """Returns ``engine_kwargs.raiden`` of a ``CheckpointEngineConfig`` (empty when unset)."""
    engine_kwargs = getattr(config, "engine_kwargs", None) or {}
    return dict(engine_kwargs.get(RAIDEN_BACKEND) or {})


def _rollout_world_size(replica: Any) -> int:
    """Number of vLLM workers (one per chip) of a rollout replica."""
    if getattr(replica, "world_size", None):
        return int(replica.world_size)
    if getattr(replica, "workers", None):
        return len(replica.workers)
    return 1


def _sampler_rank_offsets(replicas: Any) -> list[int]:
    """Global Raiden unit id of rank 0 of every rollout replica: ``replica_idx * TP`` when all replicas have the
    same TP (the running sum of the replicas' world sizes in general). Every rollout worker registers under the
    single job ``sampler`` with id ``offset + tp_rank``, so one transfer per sync addresses all replicas."""
    offsets, total = [], 0
    for replica in replicas:
        offsets.append(total)
        total += _rollout_world_size(replica)
    return offsets


REQ_ID_SUFFIX = ""  # appended to the transfer req_id (see raiden_transfer); set by the pool hook for joiner-only pushes


def _sampler_unit_offsets(manager: Any) -> list[int]:
    """Sticky global Raiden unit offset of every replica currently in ``manager.replicas``.

    A replica keeps the offset it was given the first time the manager saw it (its workers register once, under
    those ids, and ``init_raiden_sync_on_worker`` never re-registers), so a replica may leave ``manager.replicas``
    and come back without renumbering the others, and a replica that joins later takes the next free ids. With
    a fixed replica list this equals ``_sampler_rank_offsets``.
    """
    table = manager.__dict__.setdefault("_raiden_unit_offsets", {})
    for replica in manager.replicas:
        key = getattr(replica, "replica_rank", id(replica))
        if key not in table:
            table[key] = manager.__dict__.get("_raiden_next_unit", 0)
            manager._raiden_next_unit = table[key] + _rollout_world_size(replica)
    return [table[getattr(r, "replica_rank", id(r))] for r in manager.replicas]


def _get_or_start_controller(manager: Any) -> Any:
    """Returns the Raiden controller of ``manager``, starting it on the first sync."""
    controller = getattr(manager, "_raiden_controller", None)
    if controller is None:
        try:
            controller, server, address = setup_raiden_controller()
        except Exception as e:
            raise RuntimeError(f"Failed to start the RaidenControllerServer on the driver: {e}") from e
        # The server must stay referenced while the trainer and rollout workers use it.
        manager._raiden_controller, manager._raiden_server, manager._raiden_address = controller, server, address
    return controller


async def _wait_for_registration(controller: Any, src_units: list, dst_units: list, timeout_s: float) -> None:
    """Waits until every trainer (``src_units``) and rollout (``dst_units``) unit registered with the controller."""
    deadline = time.perf_counter() + timeout_s
    while True:
        # TODO(tpu): Use a public tpu_sync API to list the registered units once there is one.
        with controller._lock:
            registered = set(controller._registered_shards.keys())
        missing_src = [u for u in src_units if u not in registered]
        missing_dst = [u for u in dst_units if u not in registered]
        if not missing_src and not missing_dst:
            logger.info(
                f"[RAIDEN CONTROLLER] All {len(src_units)} Trainer and {len(dst_units)} Sampler units "
                "verified and registered."
            )
            return
        if time.perf_counter() > deadline:
            raise RuntimeError(
                f"Timeout ({timeout_s}s) waiting for workers to register with RaidenController! "
                f"Missing Trainer: {missing_src}, Missing Sampler: {missing_dst}"
            )
        await asyncio.sleep(0.1)


def _raiden_parallelism(manager: Any) -> int:
    return int(_raiden_engine_kwargs(manager.config).get("parallelism", RAIDEN_DEFAULT_PARALLELISM))


async def raiden_trainer_stage(manager: Any, global_steps: Optional[int] = None) -> float:
    """Stage 1: every trainer rank binds its tensors to its synchronizer, registers them with the controller,
    records the global shapes and stages the tensors in host memory (D2H). The trainer must hold its chips.

    ray.get blocks until all ranks finish, so it runs in a thread to keep the event loop free. Returns the
    elapsed seconds.
    """
    t0 = time.perf_counter()
    actor_refs = manager.actor_wg.update_weights(global_steps=global_steps, mode=manager.backend)
    if actor_refs is not None:
        await asyncio.to_thread(ray.get, actor_refs)
    return time.perf_counter() - t0


async def raiden_sampler_init(manager: Any) -> float:
    """Stage 2: every rollout worker allocates its receive buffers (on its chip) and registers them with the
    controller under the job ``sampler`` with its global unit id (replica offset + TP rank). A no-op on the
    workers after the first sync. Returns seconds.
    """
    t0 = time.perf_counter()
    parallelism = _raiden_parallelism(manager)
    await asyncio.gather(
        *[
            replica.server_handle.collective_rpc.remote(
                method="init_raiden_sync_on_worker",
                kwargs={"parallelism": parallelism, "rank_offset": rank_offset},
            )
            for replica, rank_offset in zip(manager.replicas, _sampler_unit_offsets(manager), strict=True)
        ]
    )
    return time.perf_counter() - t0


async def raiden_transfer(manager: Any, global_steps: Optional[int] = None, await_release: bool = True) -> dict:
    """Stages 3 and 4: registration barrier on the controller, then ONE controller-driven host-to-host transfer
    from the trainer ranks' staging buffers into the host receive buffers of every rollout worker of every
    replica, then the release of the trainer's send buffers.

    The trainer must hold its chips (its transport reads device-bound buffers); the rollout workers only
    need their host side, so they may have released theirs. With ``await_release`` the send-buffer release
    is awaited here (the colocated manager needs the trainer resident for it and releases the trainer's
    chips right after); with ``await_release=False`` the pending refs are returned under ``release_refs``
    so the disaggregated path can overlap the release with the install, as before.

    Returns ``{"barrier", "transfer", "release"}`` seconds (``release`` is 0 when not awaited).
    """
    parallelism = _raiden_parallelism(manager)
    controller = _get_or_start_controller(manager)

    # TODO(tpu): Move worker registration and barrier verification to a one-time setup step during
    # initialization (e.g. in prepare/build_process_group), since shard registration is persistent on the
    # controller and does not need to be repeated on every weight sync iteration.
    from tpu_sync.api.common import RaidenId
    from tpu_sync.rpc.raiden_controller import RaidenMemoryType

    t_barrier_start = time.perf_counter()
    src_units = [
        RaidenId(job_name="trainer", job_replica_id=str(rank), data_name="weights")
        for rank in range(manager.actor_wg.world_size)
    ]
    # All rollout workers of all replicas under the one job "sampler": ids 0 .. replicas*TP-1 (replica k holds
    # k*TP .. k*TP+TP-1), each registered with mesh [1, TP]. One transfer resharding the trainer's FSDP shards to
    # every sampler shard: one D2H staging of the send set per trainer rank per sync and one planner schedule,
    # as the in-tree code behind :v7 does; one transfer per replica staged the send set once per replica.
    # The unit ids are the sticky per-replica offsets, so only the replicas currently in ``manager.replicas``
    # are destinations (a replica that left stays registered on the controller but is not addressed).
    dst_units = [
        RaidenId(job_name="sampler", job_replica_id=str(offset + rank), data_name="weights")
        for replica, offset in zip(manager.replicas, _sampler_unit_offsets(manager), strict=True)
        for rank in range(_rollout_world_size(replica))
    ]
    logger.info(f"[RAIDEN CONTROLLER] step {global_steps}: destination sampler units {[u.job_replica_id for u in dst_units]}")
    await _wait_for_registration(controller, src_units, dst_units, timeout_s=RAIDEN_REGISTRATION_TIMEOUT_S)
    t_barrier = time.perf_counter() - t_barrier_start

    # Controller-driven P2P transfer: trainer device -> trainer host -> rollout host, resharded.
    t_transfer_start = time.perf_counter()
    transfer_future = controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        dst_mem_type=RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        expected_block_count=0,
        parallelism=parallelism,
        # The controller returns the existing (completed) future for a repeated req_id, i.e. a silent no-op; a pool
        # hook that pushes the same version a second time (joiner-only sync) sets REQ_ID_SUFFIX to keep ids unique.
        req_id=f"verl_step_{global_steps or 0}{REQ_ID_SUFFIX}",
    )
    await transfer_future.wait()
    t_transfer = time.perf_counter() - t_transfer_start

    # The trainer buffers are no longer read once the transfer is done; free them (unbind with a tpu-sync
    # that has unbind_weights, else the synchronizer teardown; see RaidenCheckpointEngine.release_sync_buffers).
    release_refs = manager.actor_wg.execute_checkpoint_engine(["release_sync_buffers"] * manager.actor_wg.world_size)
    phases: dict[str, Any] = {"barrier": t_barrier, "transfer": t_transfer, "release": 0.0}
    if await_release:
        t_release_start = time.perf_counter()
        if release_refs is not None:
            await asyncio.to_thread(ray.get, release_refs)
        phases["release"] = time.perf_counter() - t_release_start
    else:
        phases["release_refs"] = release_refs
    return phases


async def raiden_install(manager: Any, global_steps: Optional[int] = None, verify_parity: bool = False) -> dict:
    """Stage 5: every rollout worker copies the received weights to HBM and fuses them into vLLM's parameters
    (the rollout must hold its chips), the servers tag new generations with the weight version, then the
    optional parity check. Returns ``{"install", "h2d_pure"}`` (``h2d_pure`` None without worker results).
    """
    t_install_start = time.perf_counter()
    install_results = await asyncio.gather(
        *[replica.server_handle.collective_rpc.remote(method="install_raiden_weights") for replica in manager.replicas]
    )
    t_install = time.perf_counter() - t_install_start
    # collective_rpc returns one result per TP worker of each replica (when the server passes worker results
    # through); each is the timing dict returned by vLLMRaidenWorkerExtension.install_raiden_weights. The
    # slowest worker gates the sync, so report the max.
    worker_install_stats = [
        r for per_replica in install_results for r in (per_replica or []) if isinstance(r, dict) and "h2d" in r
    ]
    t_h2d_pure = max((r["h2d"] for r in worker_install_stats), default=None)
    # Tag new generations with this weight version (the trajectory staleness metrics read it), as the other
    # checkpoint backends do once the new weights are loaded. abort_replicas() already reset the prefix
    # cache, and no request has run since.
    if global_steps is not None:
        await asyncio.gather(
            *[replica.server_handle.set_global_steps.remote(global_steps) for replica in manager.replicas]
        )
    if verify_parity:
        try:
            await _verify_parity_async(manager, global_steps)
        except Exception as e:
            logger.warning(f"Failed to execute parity verification: {e}")
    return {"install": t_install, "h2d_pure": t_h2d_pure}


async def raiden_rollout_stage(manager: Any, global_steps: Optional[int] = None, verify_parity: bool = False) -> dict:
    """Stages 2 to 5 for the disaggregated case (trainer and rollout both hold their chips throughout):
    sampler registration, barrier and transfer, install overlapped with the trainer's send-buffer release.
    """
    phases: dict[str, Any] = {"init_sampler": await raiden_sampler_init(manager)}
    phases.update(await raiden_transfer(manager, global_steps, await_release=False))
    release_refs = phases.pop("release_refs", None)
    phases.update(await raiden_install(manager, global_steps, verify_parity=verify_parity))
    if release_refs is not None:
        t_release_start = time.perf_counter()
        await asyncio.to_thread(ray.get, release_refs)
        phases["release"] = time.perf_counter() - t_release_start
    return phases


async def update_raiden_weights(manager: Any, global_steps: Optional[int] = None) -> dict[str, float]:
    """Sync the trainer's weights to the vLLM rollout over Raiden, coordinated by a controller on the driver.

    Replaces ``CheckpointEngineManager.update_weights`` for ``backend=raiden`` in the disaggregated setting
    (trainer and rollout on separate slices). The colocated manager composes the same stages around its
    device hand-overs instead of calling this.

    Args:
        manager: The ``CheckpointEngineManager`` (trainer worker group, rollout replicas and config).
        global_steps: Trainer step of the weights; the rollout tags new generations with it.

    Returns:
        Per-phase timings in seconds, logged as step metrics by trainers that record sync metrics.
    """
    verify_parity = _as_bool(_raiden_engine_kwargs(manager.config).get("verify_parity", False))
    _get_or_start_controller(manager)

    t_abort_start = time.perf_counter()
    if global_steps and global_steps > 0:
        try:
            await manager.abort_replicas()
        except Exception as e:
            logger.warning(f"Failed to abort replicas at step {global_steps}: {e}")
    t_abort = time.perf_counter() - t_abort_start

    t_total_start = time.perf_counter()
    t_init_trainer = await raiden_trainer_stage(manager, global_steps)
    phases = await raiden_rollout_stage(manager, global_steps, verify_parity=verify_parity)
    t_total = time.perf_counter() - t_total_start

    logger.info(
        f"[RAIDEN TELEMETRY | Orchestrator] Step {global_steps} Completed in {t_total:.4f}s:\n"
        f"  * Sampler Quiesce/Pause  : {t_abort:.4f}s\n"
        f"  * Trainer Raiden Init    : {t_init_trainer:.4f}s\n"
        f"  * Sampler Raiden Init    : {phases['init_sampler']:.4f}s\n"
        f"  * Raiden Barrier Check   : {phases['barrier']:.4f}s\n"
        f"  * RaidenController P2P   : {phases['transfer']:.4f}s\n"
        f"  * Sampler H2D DMA        : {phases['install']:.4f}s\n"
        f"  * Total End-to-End Sync  : {t_total:.4f}s"
    )

    # Resume generation immediately
    await manager.resume_generation_replicas()

    # Surface the phase timers as step metrics. Trainers that record sync metrics (e.g. the separate-async
    # trainer's _pending_sync_metrics) log them next to timing_s/update_weights with every configured
    # backend (console / tensorboard / wandb). timing_s/update_weights ~= quiesce + total_sync.
    metrics = {
        "timing_s/tpu-sync/quiesce": t_abort,
        "timing_s/tpu-sync/trainer_init": t_init_trainer,
        "timing_s/tpu-sync/sampler_init": phases["init_sampler"],
        "timing_s/tpu-sync/barrier": phases["barrier"],
        "timing_s/tpu-sync/p2p_transfer": phases["transfer"],
        "timing_s/tpu-sync/sampler_h2d": phases["install"],
        "timing_s/tpu-sync/total_sync": t_total,
    }
    if phases.get("h2d_pure") is not None:
        # Pure _raiden_ws.h2d() time on the slowest sampler worker (sampler_h2d above also includes the
        # fuse/transpose into vLLM params, the TPU sync barrier and the RPC round trip).
        metrics["timing_s/tpu-sync/sampler_h2d_pure"] = phases["h2d_pure"]
    return metrics


async def _verify_parity_async(manager: Any, global_steps: Optional[int] = None) -> None:
    """Compare distributed norms between Trainer Rank 0 and Sampler TP workers.

    TODO(tpu): Refactor parity verification to use rank-local scalar partitioning.
    Instead of collecting per-tensor dictionaries across all Sampler workers and checking
    replicated vs sharded heuristics over 200+ tensors, each worker can reduce its model
    locally into scalar metrics (numel, l1, l2_sq) before RPC return:
      - Rank 0 accumulates both sharded tensors and replicated 1D tensors (e.g. RMSNorms).
      - Ranks 1..N-1 accumulate only sharded tensors.
    The orchestrator can then perform verification in O(ranks) pure scalar arithmetic
    rather than O(tensors * ranks) loop aggregation.
    """
    step_key = global_steps if global_steps is not None else 0
    if step_key <= 0:
        return
    # The norms of all rollout workers are summed before comparing with the trainer, which describes one model
    # copy. With several replicas (several full copies) compare each replica with the trainer on its own.
    if len(manager.replicas) > 1:
        for replica_idx, replica in enumerate(manager.replicas):
            logger.info(f"[RAIDEN PARITY] step {step_key}: checking rollout replica {replica_idx}")
            await _verify_replica_parity_async([replica], step_key, manager.actor_wg.world_size)
        return
    await _verify_replica_parity_async(list(manager.replicas), step_key, manager.actor_wg.world_size)


async def _verify_replica_parity_async(replicas: list, step_key: int, num_trainer_ranks: int = 1) -> None:
    """Compare the trainer's norms with the summed norms of the workers of ``replicas`` (one model copy).

    With the sharded export every trainer rank posts partial stats (``"ranks"``), which are summed per tensor
    once all ``num_trainer_ranks`` have posted; with the full-tensor path rank 0 posts the reference (``"master"``).
    """
    try:
        registry = get_ray_weight_registry()
        trainer_entry = None
        for _ in range(25):
            entry = await registry.get_stats.remote(step_key)
            if entry is not None and ("ranks" not in entry or len(entry["ranks"]) >= num_trainer_ranks):
                trainer_entry = entry
                break
            await asyncio.sleep(0.5)

        sampler_entries = await asyncio.gather(
            *[replica.server_handle.collective_rpc.remote(method="get_model_weights_stats") for replica in replicas]
        )

        if not trainer_entry or not sampler_entries:
            logger.warning(f"[RAIDEN PARITY] Incomplete stats data for step {step_key}")
            return

        # Unpacks and flattens the results collected from all Sampler rollout replicas into a single flat list
        # of worker dictionary objects.
        sampler_workers = [
            w
            for res in sampler_entries
            for w in (res if isinstance(res, list | tuple) else [res])
            if isinstance(w, dict)
        ]
        if not sampler_workers:
            logger.warning(
                f"[RAIDEN PARITY] No sampler worker stats for step {step_key}: the rollout server's "
                "collective_rpc must return the worker results for the parity check."
            )
            return

        if "ranks" in trainer_entry:
            trainer_master = merge_rank_stats(trainer_entry["ranks"])
        else:
            trainer_master = trainer_entry.get("master", trainer_entry)
        trainer_per_tensor = trainer_master.get("per_tensor", {})
        # Gets master list of all model tensor names (e.g., "model.layers.0.self_attn.qkv_proj.weight").
        all_param_names = list(sampler_workers[0].get("per_tensor", {}).keys()) if sampler_workers else []

        total_trainer_numel = trainer_master.get("total_numel", 0)
        total_trainer_l1 = trainer_master.get("l1_norm", 0.0)
        global_trainer_l2 = trainer_master.get("l2_norm", 0.0)

        total_sampler_l1, total_sampler_l2_sq, total_sampler_numel = 0.0, 0.0, 0
        mismatches = []

        for name in all_param_names:
            s_numels = [w.get("per_tensor", {}).get(name, {}).get("numel", 0) for w in sampler_workers]
            s_l1s = [w.get("per_tensor", {}).get(name, {}).get("l1", 0.0) for w in sampler_workers]
            s_l2_sqs = [
                w.get("per_tensor", {})
                .get(name, {})
                .get("l2_sq", w.get("per_tensor", {}).get(name, {}).get("l2", 0.0) ** 2)
                for w in sampler_workers
            ]

            is_replicated = (
                len(set(s_numels)) == 1
                and (name.endswith("layernorm.weight") or "norm" in name)
                and len(s_numels[0:1]) > 0
                and s_numels[0] < 10000
            )
            if is_replicated:
                s_agg_numel = s_numels[0]
                s_agg_l1 = s_l1s[0]
                s_agg_l2 = s_l2_sqs[0] ** 0.5
            else:
                s_agg_numel = sum(s_numels)
                s_agg_l1 = sum(s_l1s)
                s_agg_l2 = sum(s_l2_sqs) ** 0.5

            t_data = trainer_per_tensor.get(name, {})
            t_agg_numel = t_data.get("numel", 0)
            t_agg_l1 = t_data.get("l1", 0.0)

            total_sampler_numel += s_agg_numel
            total_sampler_l1 += s_agg_l1
            total_sampler_l2_sq += s_agg_l2**2

            delta_numel = abs(s_agg_numel - t_agg_numel)
            delta_l1 = abs(s_agg_l1 - t_agg_l1)
            rel_tol = 1e-3 * max(abs(t_agg_l1), 1.0)

            if delta_numel != 0 or delta_l1 > rel_tol:
                mismatches.append(f"  * {name}: Trainer(L1={t_agg_l1:.4f}) vs Sampler(L1={s_agg_l1:.4f})")

        global_sampler_l2 = total_sampler_l2_sq**0.5
        total_l1_delta = abs(total_sampler_l1 - total_trainer_l1)
        total_l2_delta = abs(global_sampler_l2 - global_trainer_l2)
        total_numel_delta = abs(total_sampler_numel - total_trainer_numel)
        total_rel_tol = 1e-3 * max(abs(total_trainer_l1), 1.0)

        if not mismatches and total_numel_delta == 0 and total_l1_delta <= total_rel_tol:
            logger.info(
                f"[RAIDEN PARITY VERIFIED | Step {step_key}] 100% DISTRIBUTED NORM PARITY CONFIRMED!\n"
                f"  * Global L1 Norm: {total_sampler_l1:.6f} "
                f"(Trainer={total_trainer_l1:.6f}, delta={total_l1_delta:.6f})\n"
                f"  * Global L2 Norm: {global_sampler_l2:.6f} "
                f"(Trainer={global_trainer_l2:.6f}, delta={total_l2_delta:.6f})\n"
                f"  * Total Parameters: {total_sampler_numel} across {len(all_param_names)} tensors"
            )
        else:
            mismatches_summary = "\n".join(mismatches[:10])
            logger.error(
                f"[RAIDEN PARITY MISMATCH | Step {step_key}] Norms do NOT match!\n"
                f"  * Trainer: numel={total_trainer_numel}, L1={total_trainer_l1:.6f}, L2={global_trainer_l2:.6f}\n"
                f"  * Sampler: numel={total_sampler_numel}, L1={total_sampler_l1:.6f}, L2={global_sampler_l2:.6f}\n"
                f"  * Mismatched Tensors ({len(mismatches)} / {len(all_param_names)}):\n"
                f"{mismatches_summary}"
            )
    except Exception as e:
        logger.warning(f"Error during parity verification for step {step_key}: {e}")
