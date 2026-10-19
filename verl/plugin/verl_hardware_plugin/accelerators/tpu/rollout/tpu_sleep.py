# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Memory-only ``sleep(level)`` / ``wake_up(tags)`` for vllm-torchtpu's ``TPUWorker``.

vLLM's GPU worker implements sleep mode with the CuMemAllocator: level 1 copies the weights to host
memory and frees the KV cache, level 2 frees both; ``wake_up(tags)`` re-maps the "weights" and
"kv_cache" allocations. vllm-torchtpu's ``TPUWorker`` has no sleep mode yet, and verl's hybrid loop
calls exactly these two engine methods around every training phase (``vLLMHttpServer._sleep_hybrid``,
``wake_up``, ``resume_kv_cache``).

This module gives ``TPUWorker`` the same two methods as pure memory operations: the tensor OBJECTS of
the model parameters, buffers and KV caches are kept (compiled graphs and modules hold them) and only
their storage is swapped for an empty tensor; ``wake_up`` re-allocates storage of the recorded shapes
and re-installs the host copies taken at level 1. No checkpoint call, no weight-transfer hooks: sleep
is a standalone memory operation. The device release of the colocated mode is a separate concern
(``platform_tpu.suspend_device``), issued by the colocated checkpoint engine manager after the sleep.

``attach_sleep_mode(TPUWorker)`` installs the methods only while the worker class lacks ``sleep``, so
this module retires itself once vllm-torchtpu ships sleep mode.
"""

import logging
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)

WEIGHTS_TAG = "weights"
KV_CACHE_TAG = "kv_cache"
_STATE_ATTR = "_verl_tpu_sleep_state"


def _worker_object(worker: Any) -> Any:
    """``collective_rpc`` targets the inner worker; keep the state on the object that owns the model runner."""
    if hasattr(worker, "model_runner"):
        return worker
    return getattr(worker, "worker", worker)


def _state(worker: Any) -> dict:
    obj = _worker_object(worker)
    state = getattr(obj, _STATE_ATTR, None)
    if state is None:
        state = {"level": 0, "weights": None, "host": {}, "kv": None, "weights_asleep": False, "kv_asleep": False}
        setattr(obj, _STATE_ATTR, state)
    return state


def _model(worker: Any) -> Any:
    obj = _worker_object(worker)
    if hasattr(obj, "get_model"):
        try:
            return obj.get_model()
        except Exception:  # noqa: BLE001 - fall back to the runner
            pass
    return getattr(getattr(obj, "model_runner", None), "model", None)


def _synchronize() -> None:
    tpu = getattr(torch, "tpu", None)
    if tpu is not None and hasattr(tpu, "synchronize"):
        try:
            tpu.synchronize()
        except Exception as e:  # noqa: BLE001
            logger.debug("torch.tpu.synchronize failed: %s", e)


def weight_tensors(model: Any) -> list[tuple[str, torch.Tensor]]:
    """``(name, tensor)`` for every device parameter and buffer of ``model``, shared tensors once.

    Non-persistent buffers (rotary caches and the like) are included: they are not in ``state_dict()``,
    so no weight loader ever rewrites them and a level-1 host copy is the only way to bring them back.
    """
    out: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if not isinstance(t, torch.Tensor) or hasattr(t, "to_local"):
            continue
        if t.numel() == 0 or t.device.type == "cpu" or id(t) in seen:
            continue
        seen.add(id(t))
        out.append((name, t))
    return out


@torch.no_grad()
def sleep(self: Any, level: int = 1) -> dict:
    """Free the KV cache storage and the weight storage (level 2) or move the weights to host (level 1).

    Returns per-worker timings and sizes in MiB. A second call while asleep is a no-op.
    """
    state = _state(self)
    if state["level"]:
        return {"skipped": True, "level": state["level"]}
    if level not in (1, 2):
        raise ValueError(f"sleep level must be 1 or 2, got {level!r}")
    obj = _worker_object(self)
    info: dict[str, Any] = {"level": level}

    # KV cache: dead across a sleep (requests are aborted and re-prefilled, the prefix cache is reset
    # on wake); record the shapes and drop the storage.
    t0 = time.perf_counter()
    kv_meta = []
    kv_bytes = 0
    kv_caches = getattr(getattr(obj, "model_runner", None), "kv_caches", None) or []
    for t in kv_caches:
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            kv_meta.append((tuple(t.shape), t.dtype, t.device))
            kv_bytes += t.numel() * t.element_size()
            t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        else:
            kv_meta.append(None)
    state["kv"] = kv_meta
    state["kv_asleep"] = True
    info["kv_drop_s"] = time.perf_counter() - t0
    info["kv_dropped_mib"] = kv_bytes / 2**20

    # Weights: level 1 keeps a host copy of every tensor; level 2 keeps only the shapes and relies on the
    # weight loader that follows the wake to rewrite the parameters.
    t1 = time.perf_counter()
    model = _model(self)
    tensors = weight_tensors(model) if model is not None else []
    state["weights"] = [(name, t, tuple(t.shape), t.dtype, t.device) for name, t in tensors]
    host_bytes = 0
    if level == 1:
        for name, t in tensors:
            state["host"][name] = t.detach().to("cpu")
            host_bytes += t.numel() * t.element_size()
        _synchronize()
    info["weight_host_copy_mib"] = host_bytes / 2**20
    weight_bytes = 0
    for _name, t, _shape, dtype, device in state["weights"]:
        weight_bytes += t.numel() * t.element_size()
        t.data = torch.empty(0, dtype=dtype, device=device)
    state["weights_asleep"] = True
    _synchronize()
    info["weight_drop_s"] = time.perf_counter() - t1
    info["weight_dropped_mib"] = weight_bytes / 2**20
    info["weight_tensors"] = len(state["weights"])
    state["level"] = level
    logger.info(
        "TPUWorker.sleep(level=%d): dropped %.0f MiB KV and %.0f MiB weights (%d tensors), %.0f MiB kept on host",
        level,
        info["kv_dropped_mib"],
        info["weight_dropped_mib"],
        len(state["weights"]),
        info["weight_host_copy_mib"],
    )
    return info


@torch.no_grad()
def wake_up(self: Any, tags: list[str] | None = None) -> dict:
    """Re-allocate what ``sleep`` dropped: ``"weights"`` first (host copies re-installed), then ``"kv_cache"``.

    ``tags=None`` wakes both. Tags whose state is already awake are no-ops.
    """
    state = _state(self)
    tags = list(tags) if tags else [WEIGHTS_TAG, KV_CACHE_TAG]
    unknown = [t for t in tags if t not in (WEIGHTS_TAG, KV_CACHE_TAG)]
    if unknown:
        raise ValueError(f"unknown wake_up tags {unknown}; expected {WEIGHTS_TAG!r} and/or {KV_CACHE_TAG!r}")
    info: dict[str, Any] = {}

    if WEIGHTS_TAG in tags and state["weights_asleep"]:
        t0 = time.perf_counter()
        realloc_bytes = 0
        for _name, t, shape, dtype, device in state["weights"] or []:
            t.data = torch.empty(shape, dtype=dtype, device=device)
            realloc_bytes += t.numel() * t.element_size()
        host_bytes = 0
        for name, t, _shape, _dtype, _device in state["weights"] or []:
            copy = state["host"].get(name)
            if copy is not None:
                t.copy_(copy)
                host_bytes += copy.numel() * copy.element_size()
        state["host"] = {}
        state["weights_asleep"] = False
        _synchronize()
        info["weight_realloc_mib"] = realloc_bytes / 2**20
        info["weight_host_restore_mib"] = host_bytes / 2**20
        info["weight_wake_s"] = time.perf_counter() - t0

    if KV_CACHE_TAG in tags and state["kv_asleep"]:
        t0 = time.perf_counter()
        obj = _worker_object(self)
        kv_caches = getattr(getattr(obj, "model_runner", None), "kv_caches", None) or []
        kv_bytes = 0
        for t, meta in zip(kv_caches, state["kv"] or [], strict=False):
            if meta is not None:
                shape, dtype, device = meta
                t.data = torch.zeros(shape, dtype=dtype, device=device)
                kv_bytes += t.numel() * t.element_size()
        state["kv_asleep"] = False
        _synchronize()
        info["kv_realloc_mib"] = kv_bytes / 2**20
        info["kv_wake_s"] = time.perf_counter() - t0

    if not state["weights_asleep"] and not state["kv_asleep"]:
        state["level"] = 0
    return info


def is_sleeping(self: Any) -> bool:
    return bool(_state(self)["level"])


def attach_sleep_mode(worker_cls: type) -> bool:
    """Give ``worker_cls`` ``sleep`` / ``wake_up`` unless vllm-torchtpu defines ``sleep`` itself.

    A ``sleep`` inherited from vLLM's generic worker base (which only raises) does not count. Returns
    True when the methods were attached. Once vllm-torchtpu implements sleep mode on ``TPUWorker`` this
    returns False and the plugin's implementation is no longer used.
    """
    if any("sleep" in vars(cls) for cls in worker_cls.__mro__ if cls.__module__.startswith("vllm_torchtpu")):
        return False
    worker_cls.sleep = sleep
    worker_cls.wake_up = wake_up
    worker_cls.is_sleeping = property(is_sleeping)
    return True
