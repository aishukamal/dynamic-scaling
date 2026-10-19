# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM worker extension that receives the trainer's weights over Raiden (tpu-sync) on TPU.

``TPUvLLMHttpServer`` passes this class to vLLM as ``worker_extension_cls`` when
``actor_rollout_ref.rollout.checkpoint_engine.backend=raiden``. The driver
(``update_raiden_weights`` in ``verl_hardware_plugin.accelerators.tpu.engines.raiden_checkpoint_engine``) calls its
methods on every vLLM worker through ``collective_rpc``:

1. ``init_raiden_sync_on_worker``: allocate TP-sharded receive buffers for the tensors the trainer
   published and register them with the Raiden controller (first sync only).
2. ``install_raiden_weights``: after the transfer, copy the received buffers to TPU HBM and fuse or
   transpose them into vLLM's parameters (every sync).
3. ``get_model_weights_stats``: weight norms for the optional parity check.

The colocated mode (trainer and rollout on one slice) adds four methods the colocated manager calls
around the device hand-overs: ``drop_receive_buffers`` / ``rebind_receive_buffers`` free and restore the
staging tensors' HBM while the synchronizer keeps its host side, and ``suspend_device`` /
``resume_device`` release and re-attach the chip through ``platform_tpu``.

This module imports vLLM (through the upstream worker extension), so only vLLM workers import it.
"""

import logging
import os
import time
from typing import Any, Optional

import ray
import torch

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from verl_hardware_plugin.accelerators.tpu.engines import raiden_checkpoint_engine as raiden
from verl_hardware_plugin.accelerators.tpu.engines.ray_weight_registry import get_ray_weight_registry
from verl_hardware_plugin.accelerators.tpu.engines.tpu_checkpoint_engine import (
    _get_parent_module,
    _resolve_model_key,
    _to_target_layout,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# How long a rollout worker waits for the controller address and the trainer's tensor shapes.
_REGISTRY_POLL_ATTEMPTS = 30
_REGISTRY_POLL_INTERVAL_S = 0.5
# Row-parallel layers: vLLM shards their input dimension (dim 1) across TP ranks; every other 2-D weight
# whose output dimension divides evenly is sharded on dim 0.
_ROW_PARALLEL_SUFFIXES = (".o_proj.weight", ".down_proj.weight")
# vLLM fuses these trainer tensors into one parameter: (fused suffix, source suffixes in concat order).
_FUSED_TARGETS = (
    (
        ".self_attn.qkv_proj.weight",
        (".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"),
    ),
    (
        ".self_attn.qkv_proj.bias",
        (".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"),
    ),
    (".mlp.gate_up_proj.weight", (".mlp.gate_proj.weight", ".mlp.up_proj.weight")),
    (".mlp.gate_up_proj.bias", (".mlp.gate_proj.bias", ".mlp.up_proj.bias")),
)


def _sampler_sharding(name: str, global_shape: list[int], tp_size: int) -> tuple[list[str], list[int]]:
    """Returns ``(sharding_spec, local_shape)`` of a trainer tensor on one vLLM TP rank."""
    spec_axes = [""] * len(global_shape)
    local_shape = list(global_shape)
    if len(global_shape) == 2:
        if any(name.endswith(s) for s in _ROW_PARALLEL_SUFFIXES):
            # Row parallel: partition input dimension (dim 1) across TP ranks
            spec_axes = ["", "tp"]
            local_shape[1] = global_shape[1] // tp_size
        elif global_shape[0] % tp_size == 0:
            # Column parallel or vocab parallel: partition output dimension (dim 0) across TP ranks
            spec_axes = ["tp", ""]
            local_shape[0] = global_shape[0] // tp_size
    return spec_axes, local_shape


class vLLMRaidenWorkerExtension(vLLMColocateWorkerExtension):
    """vLLM Worker Extension for Google Cloud TPU with Raiden P2P weight synchronization.

    Workloads using the 'raiden' checkpoint engine for high-speed TPU weight sync
    must use this worker extension to bind TPU model tensors with WeightSynchronizer
    and orchestrate dynamic zero-copy H2D DMA transfers during rollout.
    """

    def _get_vllm_model(self) -> Any:
        """Extract the underlying nn.Module from the vLLM V1 worker."""
        worker: Any = getattr(self, "worker", self)
        return worker.model_runner.model

    def _wait_for_raiden_registry(self, rank_val: int) -> tuple[str, dict[str, list[int]]]:
        """Returns the controller address and the trainer's global tensor shapes from ``RayWeightRegistry``."""
        registry = get_ray_weight_registry()
        for _ in range(_REGISTRY_POLL_ATTEMPTS):
            try:
                addr, shapes = ray.get([registry.get_controller_address.remote(), registry.get_global_shapes.remote()])
                if addr and shapes:
                    return addr, shapes
            except Exception as e:
                logger.debug(f"Raiden Sampler Rank {rank_val}: registry lookup failed: {e}")
            time.sleep(_REGISTRY_POLL_INTERVAL_S)
        raise RuntimeError(
            f"Raiden Sampler Rank {rank_val}: No RaidenController address or "
            f"global shapes found in RayWeightRegistry after timeout"
        )

    def init_raiden_sync_on_worker(
        self, parallelism: int = raiden.RAIDEN_DEFAULT_PARALLELISM, rank_offset: int = 0
    ) -> bool:
        """Initialize Raiden WeightSynchronizer listener and register with central RaidenController.

        Only the first call does work: the receive buffers, the synchronizer and the registration are
        reused by every later sync. Every rollout worker of every replica registers under the single Raiden
        job ``sampler`` with the global unit id ``rank_offset + rank`` (``rank`` is the TP rank inside this
        replica; the driver passes ``rank_offset = replica_idx * TP``), so the controller sees one job of
        ``replicas * TP`` units with mesh ``[1, TP]`` each and one transfer per sync covers all replicas.
        """
        if getattr(self, "_raiden_ws", None) is not None:
            return True

        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            logger.warning("Raiden Sampler: could not locate vllm_model to bind parameters.")
            return False

        from tpu_sync.rpc import raiden_controller, raiden_service_pb2

        bind_ip = ray.util.get_node_ip_address().strip("[]")
        # ``rank`` is the TP rank inside this replica; the global Raiden unit id adds the replica's offset.
        rank_val = int(rank_offset) + int(getattr(self, "rank", 0))
        # Let the OS pick the control listener port, as the trainer side does (a fixed base + rank collides as
        # soon as two replicas share a host). The port actually bound is read back from the synchronizer and
        # registered with the controller below.
        listener_port = 0

        # 1. Fetch RaidenController address and un-fused global shapes from RayWeightRegistry
        controller_addr, global_shapes_map = self._wait_for_raiden_registry(rank_val)

        # 2. Determine tensor parallel size from vLLM V1 config
        worker: Any = getattr(self, "worker", self)
        parallel_config = getattr(getattr(worker, "vllm_config", None), "parallel_config", None)
        tp_size = getattr(parallel_config, "tensor_parallel_size", None)
        if not tp_size or tp_size <= 0:
            raise RuntimeError(
                f"Raiden Sampler Rank {rank_val}: Unable to determine tensor_parallel_size (tp_size) "
                f"from worker.vllm_config.parallel_config!"
            )

        # 3. Allocate local TPU staging buffers matching Trainer un-fused layout and sharding specs
        # TODO(tpu): Copy the received weights straight into vLLM's parameters, with no staging copy. The
        # staging tensors are a second full copy of this rank's TP shard (~1.9 GiB per chip for Qwen3-8B,
        # ~7.6 GiB for Qwen3-32B at TP=8) that must fit next to vLLM's preallocated weights and KV cache on
        # every sync. Unfused tensors (o_proj, down_proj, norms, embed_tokens, lm_head) could bind the vLLM
        # parameter buffers directly; fused qkv_proj / gate_up_proj and transposed (_tpu_weight_flipped)
        # weights need tpu_sync to write into a slice or layout of the target tensor.
        staging_tensors = {}
        variable_protos = []
        valid_params = []

        for idx, (name, g_shape) in enumerate(sorted(global_shapes_map.items(), key=lambda x: x[0])):
            g_shape = list(g_shape)
            spec_axes, local_shape = _sampler_sharding(name, g_shape, tp_size)
            sharding_mesh = [tp_size if axis == "tp" else 1 for axis in spec_axes]

            # Allocate a local staging buffer with matching layout (the trainer sends bf16)
            t = torch.empty(local_shape, dtype=torch.bfloat16, device=torch.device(raiden.RAIDEN_DEVICE))
            staging_tensors[name] = t
            valid_params.append((name, t))

            variable_protos.append(
                raiden_service_pb2.VariableMetadataProto(
                    name=name,
                    shape=g_shape,
                    mesh_shape=sharding_mesh,
                    layout=list(range(len(local_shape) - 1, -1, -1)),
                    item_size=t.element_size(),
                    layer_idx=idx,
                    sharding_spec=spec_axes,
                )
            )

        self._raiden_staging = staging_tensors
        self._sorted_vllm_params = valid_params

        raiden.tpu_synchronize()

        logger.info(
            f"Raiden Sampler Rank {rank_val}: binding {len(valid_params)} staging tensors to WeightSynchronizer "
            f"(sample: name={valid_params[0][0]}, shape={valid_params[0][1].shape}, "
            f"total numel={sum(t.numel() for _, t in valid_params)})"
        )

        self._raiden_ws = raiden.create_torch_weight_synchronizer(
            [[t] for _, t in valid_params],
            local_port=0,
            parallelism=parallelism,
            listener_port=listener_port,
            bind_ip=bind_ip,
        )

        # Whether a tensor can skip the CPU (de)tiling pass is decided by Raiden's planner per transfer, from the
        # SOURCE and destination shards together, and delivered to this listener with the transfer. Deciding it
        # here from the local shape alone made the two sides disagree whenever a trainer shard was not 8-row
        # aligned (Qwen3-14B lm_head / embed_tokens at FSDP dp=32: 4748 rows): the sender de-tiled to row-major
        # and h2d() copied those bytes raw into tiled HBM, permuting the two vocabulary tensors while every norm
        # stayed the same.

        try:
            ctrl_client = raiden_controller.RaidenControllerClientFacade(controller_addr)
            unit_id = raiden_controller.RaidenId("sampler", str(rank_val), "weights")
            ctrl_client.register_work_unit(
                unit_id,
                [f"{bind_ip}:{self._raiden_ws.local_port}"],
                f"{bind_ip}:{self._raiden_ws.listener_port}",
                mesh_shape=[1, tp_size],
                variables=variable_protos,
                mesh_axes=["fsdp", "tp"],
            )
        except Exception as e:
            logger.error(f"Raiden Sampler Rank {rank_val} failed to register with RaidenController: {e}")
            raise
        logger.info(
            f"Raiden Sampler Rank {rank_val} bound {len(valid_params)} dynamic staging tensors and registered "
            f"directly with RaidenController ({controller_addr}): mesh_shape=[1, {tp_size}], "
            f"data_port={self._raiden_ws.local_port}, listener_port={self._raiden_ws.listener_port}"
        )

        return True

    @torch.no_grad()
    def install_raiden_weights(self) -> dict[str, float]:
        """Install received weights from host staging buffer into TPU HBM via zero-copy H2D DMA
        and fuse/transpose them directly into vLLM model parameters.

        Returns:
            Per-worker timings in seconds: ``total`` (whole install), ``h2d`` (pure ``_raiden_ws.h2d()``)
            and ``sync`` (final TPU sync barrier). Empty if the synchronizer is not initialized.
        """
        if getattr(self, "_raiden_ws", None) is None:
            logger.warning("Raiden Sampler: install_raiden_weights called before _raiden_ws was initialized.")
            return {}

        t_start = time.perf_counter()
        t_h2d_start = time.perf_counter()
        self._raiden_ws.h2d()
        t_h2d = time.perf_counter() - t_h2d_start

        # Unpack, fuse, and transpose staging tensors into vLLM model parameters
        vllm_model = self._get_vllm_model()
        model_sd = vllm_model.state_dict() if hasattr(vllm_model, "state_dict") else vllm_model.model.state_dict()
        module_dict = dict(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else {}

        def resolve_model_key(k: str) -> Optional[str]:
            resolved = _resolve_model_key(k, model_sd)
            return resolved if resolved in model_sd else None

        def copy_into(resolved_key: str, src: torch.Tensor) -> None:
            target_param = model_sd[resolved_key]
            target_local = target_param.to_local() if hasattr(target_param, "to_local") else target_param
            parent_mod = _get_parent_module(resolved_key, module_dict)
            is_flipped = bool(getattr(parent_mod, "_tpu_weight_flipped", False))
            target_local.copy_(_to_target_layout(src, target_local, is_flipped))

        consumed_staging = set()

        # 1. Handle fused projections (QKV and Gate-Up)
        for target_suffix, src_suffixes in _FUSED_TARGETS:
            primary_src = src_suffixes[0]
            for name in list(self._raiden_staging.keys()):
                if primary_src in name:
                    layer_src_keys = [name.replace(primary_src, s) for s in src_suffixes]
                    if all(sk in self._raiden_staging for sk in layer_src_keys):
                        resolved_key = resolve_model_key(name.replace(primary_src, target_suffix))
                        if resolved_key is not None:
                            fused = torch.cat([self._raiden_staging[sk] for sk in layer_src_keys], dim=0)
                            copy_into(resolved_key, fused)
                            consumed_staging.update(layer_src_keys)

        # 2. Handle all remaining non-fused parameters (o_proj, down_proj, layernorms, embeddings)
        for name, src_t in self._raiden_staging.items():
            if name in consumed_staging:
                continue
            resolved_key = resolve_model_key(name)
            if resolved_key is not None:
                copy_into(resolved_key, src_t)

        # 3. Handle tied word embeddings. Only when the trainer did not send lm_head itself: for untied
        #    models (e.g. Qwen3-8B / 32B) the received lm_head must not be overwritten by embed_tokens.
        received_lm_head = any(k == "lm_head.weight" or k.endswith(".lm_head.weight") for k in self._raiden_staging)
        if received_lm_head:
            pass
        elif (
            hasattr(vllm_model, "lm_head")
            and hasattr(vllm_model, "model")
            and hasattr(vllm_model.model, "embed_tokens")
        ):
            if vllm_model.lm_head.weight.data_ptr() != vllm_model.model.embed_tokens.weight.data_ptr():
                vllm_model.lm_head.weight.copy_(vllm_model.model.embed_tokens.weight)
        elif hasattr(vllm_model, "lm_head") and hasattr(vllm_model, "embed_tokens"):
            if vllm_model.lm_head.weight.data_ptr() != vllm_model.embed_tokens.weight.data_ptr():
                vllm_model.lm_head.weight.copy_(vllm_model.embed_tokens.weight)

        t_sync_start = time.perf_counter()
        raiden.tpu_synchronize()
        t_sync = time.perf_counter() - t_sync_start
        t_total = time.perf_counter() - t_start

        logger.info(
            f"[RAIDEN TELEMETRY | Sampler Worker] Sampler Rank {getattr(self, 'rank', 0)}: install_raiden_weights "
            f"completed in {t_total:.4f}s (H2D={t_h2d:.4f}s, TPUSyncBarrier={t_sync:.4f}s)"
        )
        # Returned through collective_rpc so the orchestrator can log them as step metrics.
        return {"total": t_total, "h2d": t_h2d, "sync": t_sync}

    # ------------------------------------------------------------------
    # Colocated mode: the receive buffers across a device release, and the device release itself.
    # The colocated checkpoint engine manager calls these through collective_rpc; nothing else does.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def drop_receive_buffers(self) -> dict:
        """Free the HBM of the Raiden staging tensors before this worker releases the device.

        The staging tensors are the second bf16 copy of this rank's TP shard that ``install_raiden_weights``
        copies into vLLM's parameters. ``WeightSynchronizer.unbind_weights()`` drops the synchronizer's holds
        on them while keeping its host receive buffers, listener and controller registration, so the
        controller's transfer into this worker's host memory still lands while the device is released;
        then the tensors' storage is swapped for an empty one (the objects stay). Returns timings and the
        size dropped in MiB. A no-op before the first registration, and with a tpu-sync build without
        ``unbind_weights`` (logged once; the staging then stays in the suspended image).
        """
        ws = getattr(self, "_raiden_ws", None)
        staging = getattr(self, "_raiden_staging", None)
        if ws is None or not staging:
            return {"skipped": True, "reason": "no receive buffers"}
        if getattr(self, "_receive_buffers_dropped", False):
            return {"skipped": True, "reason": "already dropped"}
        if not hasattr(ws, "unbind_weights"):
            if not getattr(self, "_warned_no_unbind", False):
                logger.warning(
                    "Raiden staging release skipped: this tpu_sync WeightSynchronizer has no unbind_weights(); "
                    "the receive buffers stay resident across the device release."
                )
                self._warned_no_unbind = True
            return {"skipped": True, "reason": "no unbind_weights"}
        t0 = time.perf_counter()
        if getattr(self, "_receive_buffer_meta", None) is None:
            self._receive_buffer_meta = [(name, tuple(t.shape), t.dtype, t.device) for name, t in staging.items()]
        ws.unbind_weights()
        t1 = time.perf_counter()
        nbytes = 0
        for t in staging.values():
            nbytes += t.numel() * t.element_size()
            t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        raiden.tpu_synchronize()
        self._receive_buffers_dropped = True
        info = {
            "unbind_s": t1 - t0,
            "drop_s": time.perf_counter() - t1,
            "dropped_mib": nbytes / 2**20,
            "tensors": len(staging),
        }
        logger.info(
            f"Raiden Sampler Rank {getattr(self, 'rank', 0)}: dropped {info['dropped_mib']:.0f} MiB of receive "
            f"buffers ({info['tensors']} tensors) in {info['unbind_s'] + info['drop_s']:.3f}s"
        )
        return info

    @torch.no_grad()
    def rebind_receive_buffers(self) -> dict:
        """Re-allocate the staging tensors dropped by ``drop_receive_buffers`` and bind them again.

        Fresh storage of the recorded shapes is given to the same tensor objects, in the registration
        order of ``_sorted_vllm_params``, and ``bind_weights`` re-attaches them to the synchronizer so the
        next ``install_raiden_weights`` h2d() lands as before. The skip-tiling decision is the planner's
        and arrives with each transfer, so nothing is re-applied here.
        """
        if not getattr(self, "_receive_buffers_dropped", False):
            return {"skipped": True, "reason": "not dropped"}
        ws = self._raiden_ws
        staging = self._raiden_staging
        t0 = time.perf_counter()
        nbytes = 0
        for name, shape, dtype, device in self._receive_buffer_meta:
            t = staging[name]
            t.data = torch.empty(shape, dtype=dtype, device=device)
            nbytes += t.numel() * t.element_size()
        raiden.tpu_synchronize()
        t1 = time.perf_counter()
        ws.bind_weights([[t] for _, t in self._sorted_vllm_params])
        self._receive_buffers_dropped = False
        info = {"realloc_s": t1 - t0, "rebind_s": time.perf_counter() - t1, "realloc_mib": nbytes / 2**20}
        logger.info(
            f"Raiden Sampler Rank {getattr(self, 'rank', 0)}: re-allocated and rebound {info['realloc_mib']:.0f} MiB "
            f"of receive buffers in {info['realloc_s'] + info['rebind_s']:.3f}s"
        )
        return info

    def suspend_device(self) -> dict:
        """Release this vLLM worker's TPU chip (``platform_tpu.suspend_device``); idempotent."""
        from verl_hardware_plugin.accelerators.tpu import platform_tpu

        return platform_tpu.suspend_device()

    def resume_device(self) -> dict:
        """Re-attach this vLLM worker's TPU chip (``platform_tpu.resume_device``); idempotent."""
        from verl_hardware_plugin.accelerators.tpu import platform_tpu

        return platform_tpu.resume_device()

    def get_model_weights_stats(self) -> dict:
        """Computes deterministic parameter count, L1 norm, and L2 norm of this worker's received weights."""
        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            return {"error": "No model found on worker"}

        if not getattr(self, "_sorted_vllm_params", None):
            raise RuntimeError(
                f"get_model_weights_stats is only supported for Raiden after initialization, "
                f"but self._sorted_vllm_params is not initialized on worker rank {getattr(self, 'rank', 'unknown')}!"
            )

        res = raiden.compute_tensor_stats(self._sorted_vllm_params)
        res["rank"] = getattr(self, "rank", 0)
        return res
