# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the memory-only TPUWorker sleep / wake_up (``rollout/tpu_sleep.py``).

The fake worker holds a small model and KV caches on a fake "device": CPU tensors whose device is
patched to look non-CPU, so the module's "skip host tensors" rule does not exclude them.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from verl_hardware_plugin.accelerators.tpu.rollout import tpu_sleep


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 8, bias=False)
        self.head = nn.Linear(8, 4, bias=False)
        self.tied = nn.Linear(4, 8, bias=False)
        self.tied.weight = self.embed.weight  # shared parameter: counted once
        self.register_buffer("rotary", torch.arange(6.0), persistent=False)


class _Worker:
    """Stands in for vllm-torchtpu's TPUWorker: ``model_runner.model`` and ``model_runner.kv_caches``."""

    def __init__(self):
        self.model_runner = SimpleNamespace(
            model=_Model(), kv_caches=[torch.ones(2, 3), torch.ones(4), "not a tensor", torch.empty(0)]
        )


class _Wrapper:
    """vLLM's RayWorkerWrapper shape: the inner worker owns the runner."""

    def __init__(self, worker):
        self.worker = worker


@pytest.fixture(autouse=True)
def device_looks_like_tpu(monkeypatch):
    # weight_tensors skips tensors whose device.type is "cpu"; treat every tensor as a device tensor here.
    monkeypatch.setattr(
        tpu_sleep, "weight_tensors", lambda model: _dedup(list(model.named_parameters()) + list(model.named_buffers()))
    )


def _dedup(items):
    out, seen = [], set()
    for name, t in items:
        if id(t) in seen or t.numel() == 0:
            continue
        seen.add(id(t))
        out.append((name, t))
    return out


def _fill(worker):
    model = worker.model_runner.model
    with torch.no_grad():
        model.embed.weight.copy_(torch.arange(32.0).reshape(8, 4))
        model.rotary.copy_(torch.arange(6.0) * 10)
    return {name: t.clone() for name, t in _dedup(list(model.named_parameters()) + list(model.named_buffers()))}


def _numels(worker):
    model = worker.model_runner.model
    return {name: t.numel() for name, t in list(model.named_parameters()) + list(model.named_buffers())}


def test_level_2_drops_weights_and_kv_and_wake_reallocates_in_order():
    worker = _Worker()
    before = _fill(worker)
    info = tpu_sleep.sleep(worker, level=2)

    assert info["level"] == 2
    assert info["kv_dropped_mib"] == pytest.approx((6 + 4) * 4 / 2**20)
    assert info["weight_host_copy_mib"] == 0
    assert info["weight_tensors"] == 3  # embed (shared with tied), head and the rotary buffer
    assert all(n == 0 for n in _numels(worker).values())
    assert worker.model_runner.kv_caches[0].numel() == 0 and worker.model_runner.kv_caches[1].numel() == 0
    assert worker.model_runner.kv_caches[2] == "not a tensor"
    assert tpu_sleep.is_sleeping(worker)

    # Weights first: a loader can rewrite them before the KV cache exists again.
    info = tpu_sleep.wake_up(worker, ["weights"])
    assert "weight_realloc_mib" in info and "kv_realloc_mib" not in info
    assert _numels(worker) == {name: t.numel() for name, t in before.items()}
    assert worker.model_runner.model.tied.weight is worker.model_runner.model.embed.weight  # sharing survives
    assert worker.model_runner.kv_caches[0].numel() == 0
    assert tpu_sleep.is_sleeping(worker)  # the KV cache is still asleep

    info = tpu_sleep.wake_up(worker, ["kv_cache"])
    assert info["kv_realloc_mib"] == pytest.approx((6 + 4) * 4 / 2**20)
    assert torch.equal(worker.model_runner.kv_caches[0], torch.zeros(2, 3))
    assert torch.equal(worker.model_runner.kv_caches[1], torch.zeros(4))
    assert not tpu_sleep.is_sleeping(worker)


def test_level_1_keeps_host_copies_and_reinstalls_them():
    worker = _Wrapper(_Worker())  # the wrapper shape: state lives on the inner worker
    before = _fill(worker.worker)
    info = tpu_sleep.sleep(worker, level=1)
    assert info["weight_host_copy_mib"] == pytest.approx((32 + 32 + 6) * 4 / 2**20)
    assert worker.worker.model_runner.model.embed.weight.numel() == 0

    # Both tags in one call: weights are processed before the KV cache.
    order = []
    original_empty, original_zeros = torch.empty, torch.zeros
    with (
        mock.patch.object(
            torch, "empty", side_effect=lambda *a, **k: (order.append("weights"), original_empty(*a, **k))[1]
        ),
        mock.patch.object(torch, "zeros", side_effect=lambda *a, **k: (order.append("kv"), original_zeros(*a, **k))[1]),
    ):
        info = tpu_sleep.wake_up(worker)
    assert order and order.index("kv") > order.index("weights") and "weights" not in order[order.index("kv") :]
    assert info["weight_host_restore_mib"] == pytest.approx((32 + 32 + 6) * 4 / 2**20)
    model = worker.worker.model_runner.model
    assert torch.equal(model.embed.weight, before["embed.weight"])
    assert torch.equal(model.rotary, before["rotary"])
    assert not tpu_sleep.is_sleeping(worker)
    assert getattr(worker.worker, tpu_sleep._STATE_ATTR)["host"] == {}  # host copies released after the wake


def test_repeated_calls_and_bad_arguments():
    worker = _Worker()
    _fill(worker)
    with pytest.raises(ValueError, match="level"):
        tpu_sleep.sleep(worker, level=3)
    assert tpu_sleep.sleep(worker, level=2)["level"] == 2
    assert tpu_sleep.sleep(worker, level=1) == {"skipped": True, "level": 2}
    with pytest.raises(ValueError, match="unknown wake_up tags"):
        tpu_sleep.wake_up(worker, ["kv"])
    tpu_sleep.wake_up(worker)
    assert tpu_sleep.wake_up(worker) == {}  # already awake: nothing re-allocated
    assert tpu_sleep.wake_up(worker, ["kv_cache"]) == {}


def test_attach_only_when_vllm_torchtpu_has_no_sleep():
    class WorkerBase:  # vLLM's generic base may define a raising sleep; it does not count
        def sleep(self, level=1):
            raise NotImplementedError

    WorkerBase.__module__ = "vllm.v1.worker.worker_base"

    class TPUWorker(WorkerBase):
        pass

    TPUWorker.__module__ = "vllm_torchtpu.worker.tpu_worker"
    assert tpu_sleep.attach_sleep_mode(TPUWorker) is True
    assert TPUWorker.sleep is tpu_sleep.sleep and TPUWorker.wake_up is tpu_sleep.wake_up
    assert tpu_sleep.attach_sleep_mode(TPUWorker) is False  # now it has one of its own (ours)

    class NativeTPUWorker:
        def sleep(self, level=1):
            return "native"

    NativeTPUWorker.__module__ = "vllm_torchtpu.worker.tpu_worker"
    assert tpu_sleep.attach_sleep_mode(NativeTPUWorker) is False
    assert NativeTPUWorker().sleep() == "native"
