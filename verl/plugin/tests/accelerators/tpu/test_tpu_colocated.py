# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the TPU colocated checkpoint engine manager: the exact order of the hand-over calls.

The Raiden stages are replaced by recorders, the trainer worker group and the rollout replicas are
in-process fakes; no Ray, no tpu-sync, no device.
"""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.workers.config import CheckpointEngineConfig
from verl.workers.rollout.replica import RolloutMode
from verl_hardware_plugin.accelerators.tpu import platform_tpu
from verl_hardware_plugin.accelerators.tpu.engines import tpu_colocated


def _done(value=None):
    async def coro():
        return value

    return coro()


class _FakeWorker:
    """A trainer worker actor: ``__ray_call__.remote(fn)`` runs ``fn`` here and records its name."""

    def __init__(self, events, rank):
        self.events, self.rank = events, rank
        self.__ray_call__ = SimpleNamespace(remote=self._call)

    def _call(self, fn):
        self.events.append(f"T{self.rank}:{fn.__name__}")
        return _done(fn(self))


class _FakeServer:
    def __init__(self, events, name):
        self.events, self.name = events, name
        self.collective_rpc = SimpleNamespace(remote=self._rpc)
        self.wake_up = SimpleNamespace(remote=self._wake_up)
        self.set_global_steps = SimpleNamespace(remote=self._set_global_steps)

    def _rpc(self, method, args=(), kwargs=None, timeout=None):
        self.events.append(f"{self.name}:rpc:{method}")
        return _done([{"dropped_mib": 3.0}] if method == "drop_receive_buffers" else [{}])

    def _wake_up(self, tags=None):
        self.events.append(f"{self.name}:wake_up:{','.join(tags)}")
        return _done()

    def _set_global_steps(self, step):
        self.events.append(f"{self.name}:set_global_steps:{step}")
        return _done()


class _FakeReplica:
    def __init__(self, events, idx, backend="raiden", mode=RolloutMode.HYBRID):
        self.events, self.idx = events, idx
        self.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(backend=backend))
        self.rollout_mode = mode
        self.servers = [_FakeServer(events, f"S{idx}")]
        self.server_handle = self.servers[0]
        self.world_size = 2

    async def sleep(self):
        self.events.append(f"S{self.idx}:sleep")

    async def wake_up(self):
        self.events.append(f"S{self.idx}:wake_up")

    async def abort_all_requests(self, reject_request=False):
        self.events.append(f"S{self.idx}:abort")

    async def resume_generation(self):
        self.events.append(f"S{self.idx}:resume_generation")

    async def resume_kv_cache(self):
        self.events.append(f"S{self.idx}:resume_kv_cache")


@pytest.fixture
def world(monkeypatch):
    events: list[str] = []
    calls = {"suspend": 0, "resume": 0}

    def suspend():
        calls["suspend"] += 1
        return {"suspend_s": 0.5, "rss_delta_mib": 100.0}

    def resume():
        calls["resume"] += 1
        return {"resume_s": 0.7}

    monkeypatch.setattr(platform_tpu, "suspend_device", suspend)
    monkeypatch.setattr(platform_tpu, "resume_device", resume)

    async def trainer_stage(manager, global_steps=None):
        events.append(f"stage:trainer({global_steps})")
        return 1.0

    async def sampler_init(manager):
        events.append("stage:sampler_init")
        return 2.0

    async def transfer(manager, global_steps=None, await_release=True):
        events.append(f"stage:transfer(await_release={await_release})")
        return {"barrier": 0.1, "transfer": 0.2, "release": 0.3}

    async def install(manager, global_steps=None, verify_parity=False):
        events.append(f"stage:install(parity={verify_parity})")
        for replica in manager.replicas:
            replica.events.append(f"S{replica.idx}:set_global_steps:{global_steps}")
        return {"install": 0.4, "h2d_pure": 0.05}

    monkeypatch.setattr(tpu_colocated, "raiden_trainer_stage", trainer_stage)
    monkeypatch.setattr(tpu_colocated, "raiden_sampler_init", sampler_init)
    monkeypatch.setattr(tpu_colocated, "raiden_transfer", transfer)
    monkeypatch.setattr(tpu_colocated, "raiden_install", install)
    monkeypatch.setattr(tpu_colocated, "_get_or_start_controller", lambda manager: events.append("controller"))

    actor_wg = SimpleNamespace(workers=[_FakeWorker(events, r) for r in range(2)], world_size=2)
    replicas = [_FakeReplica(events, i) for i in range(2)]
    return SimpleNamespace(events=events, calls=calls, actor_wg=actor_wg, replicas=replicas)


def _manager(world, backend_in_config="naive", engine_kwargs=None):
    config = CheckpointEngineConfig(backend=backend_in_config, engine_kwargs=engine_kwargs or {})
    return tpu_colocated.TPUColocatedCheckpointEngineManager(config, world.actor_wg, world.replicas)


def test_init_rereads_the_configured_backend_and_rejects_others(world):
    manager = _manager(world)  # trainer_base forced "naive"; the replicas' rollout config says raiden
    assert manager.backend == "raiden" and manager.config.backend == "raiden"
    assert manager._trainer_suspended and not manager._samplers_suspended
    with pytest.raises(ValueError, match="backend"):
        tpu_colocated.TPUColocatedCheckpointEngineManager(
            CheckpointEngineConfig(backend="nccl"), world.actor_wg, world.replicas
        )


def test_sleep_replicas_order_and_abort_noop_while_suspended(world):
    manager = _manager(world)
    world.events.clear()
    manager.sleep_replicas()
    assert world.events == [
        "S0:sleep",
        "S1:sleep",
        "S0:rpc:drop_receive_buffers",
        "S1:rpc:drop_receive_buffers",
        "S0:rpc:suspend_device",
        "S1:rpc:suspend_device",
        "T0:_resume_trainer_worker",
        "T1:_resume_trainer_worker",
    ]
    assert world.calls == {"suspend": 0, "resume": 2}  # the extension's suspend is a fake RPC; the trainer resumed
    assert manager._samplers_suspended and not manager._trainer_suspended

    world.events.clear()
    manager.abort_replicas()
    assert world.events == []  # nothing is in flight without chips
    manager.sleep_replicas()
    assert world.events == []  # idempotent


def test_update_weights_first_sync_round_trip_then_regular_order(world):
    manager = _manager(world, engine_kwargs={"raiden": {"verify_parity": True}})
    manager.sleep_replicas()  # the loop's setup call: rollout released, trainer back
    world.events.clear()

    metrics = manager.update_weights(0)

    first_round_trip = [
        "controller",
        "stage:trainer(0)",
        "T0:_suspend_trainer_worker",
        "T1:_suspend_trainer_worker",
        "S0:rpc:resume_device",
        "S1:rpc:resume_device",
        "stage:sampler_init",
        "S0:rpc:drop_receive_buffers",
        "S1:rpc:drop_receive_buffers",
        "S0:rpc:suspend_device",
        "S1:rpc:suspend_device",
        "T0:_resume_trainer_worker",
        "T1:_resume_trainer_worker",
    ]
    regular = [
        "stage:trainer(0)",
        "stage:transfer(await_release=True)",
        "T0:_suspend_trainer_worker",
        "T1:_suspend_trainer_worker",
        "S0:rpc:resume_device",
        "S1:rpc:resume_device",
        "S0:wake_up:weights",
        "S1:wake_up:weights",
        "S0:rpc:rebind_receive_buffers",
        "S1:rpc:rebind_receive_buffers",
        "stage:install(parity=True)",
        "S0:set_global_steps:0",
        "S1:set_global_steps:0",
        "S0:resume_kv_cache",
        "S1:resume_kv_cache",
    ]
    assert world.events == first_round_trip + regular
    # Mode-neutral: no resume_generation from the manager (the trainer that paused resumes; sync never pauses).
    assert not any(e.endswith(":resume_generation") for e in world.events)
    assert manager._trainer_suspended and not manager._samplers_suspended
    assert metrics["timing_s/tpu-sync/sampler_init"] == 2.0
    assert metrics["timing_s/tpu-sync/trainer_init_first"] == 1.0
    assert metrics["timing_s/tpu-sync/p2p_transfer"] == 0.2
    assert metrics["timing_s/tpu-sync/sampler_h2d_pure"] == 0.05
    for key in ("trainer_suspend", "sampler_resume", "wake_weights", "sampler_rebind", "wake_kv", "sampler_sleep"):
        assert f"timing_s/tpu-colocated/{key}" in metrics, key

    # Second step: no round trip, the setup sleep's timings come with this sync's metrics.
    manager.sleep_replicas()
    world.events.clear()
    metrics = manager.update_weights(1)
    assert world.events == ["controller"] + [e.replace("(0)", "(1)").replace(":0", ":1") for e in regular]
    assert "timing_s/tpu-sync/sampler_init" not in metrics
    assert "timing_s/tpu-colocated/sampler_drop" in metrics


def test_update_weights_from_a_suspended_trainer_swaps_in_first(world):
    manager = _manager(world)  # fresh after init_hybrid: trainer released, rollout resident
    world.events.clear()
    manager.update_weights(0)
    assert world.events[:8] == [
        "S0:sleep",
        "S1:sleep",
        "S0:rpc:drop_receive_buffers",
        "S1:rpc:drop_receive_buffers",
        "S0:rpc:suspend_device",
        "S1:rpc:suspend_device",
        "T0:_resume_trainer_worker",
        "T1:_resume_trainer_worker",
    ]


def test_update_weights_plasma_backend(world):
    for replica in world.replicas:
        replica.config.checkpoint_engine.backend = "tpu"
    refs = []
    world.actor_wg.update_weights = lambda global_steps=None, mode=None: refs.append((global_steps, mode)) or None
    manager = _manager(world)
    manager.sleep_replicas()
    world.events.clear()
    manager.update_weights(3)
    assert refs == [(3, "tpu")]
    assert world.events == [
        "T0:_suspend_trainer_worker",
        "T1:_suspend_trainer_worker",
        "S0:rpc:resume_device",
        "S1:rpc:resume_device",
        "S0:wake_up:weights",
        "S1:wake_up:weights",
        "S0:rpc:load_weights_from_ray_registry",
        "S1:rpc:load_weights_from_ray_registry",
        "S0:set_global_steps:3",
        "S1:set_global_steps:3",
        "S0:resume_kv_cache",
        "S1:resume_kv_cache",
    ]


def test_wake_up_replicas_hands_over_and_wakes_both(world):
    manager = _manager(world)
    manager.sleep_replicas()
    world.events.clear()
    manager.wake_up_replicas()
    assert world.events == [
        "T0:_suspend_trainer_worker",
        "T1:_suspend_trainer_worker",
        "S0:rpc:resume_device",
        "S1:rpc:resume_device",
        "S0:wake_up:weights,kv_cache",
        "S1:wake_up:weights,kv_cache",
        "S0:rpc:rebind_receive_buffers",
        "S1:rpc:rebind_receive_buffers",
    ]


# ---------------------------------------------------------------------------
# Selection without C1
# ---------------------------------------------------------------------------


def test_select_checkpoint_manager_picks_the_tpu_manager_for_tpu_hybrid_raiden(world, monkeypatch):
    platform = SimpleNamespace(device_name="tpu")
    monkeypatch.setattr("verl.plugin.platform.get_platform", lambda: platform)
    config = CheckpointEngineConfig(backend="naive")
    manager = tpu_colocated.select_checkpoint_manager(config, world.actor_wg, world.replicas)
    assert isinstance(manager, tpu_colocated.TPUColocatedCheckpointEngineManager) and manager.backend == "raiden"

    # Not hybrid (standalone replicas), not TPU, or a GPU backend: the stock manager, backend untouched.
    standalone = [_FakeReplica(world.events, 0, mode=RolloutMode.STANDALONE)]
    stock = tpu_colocated.select_checkpoint_manager(CheckpointEngineConfig(backend="naive"), world.actor_wg, standalone)
    assert type(stock) is CheckpointEngineManager and stock.backend == "naive"
    platform.device_name = "cuda"
    stock = tpu_colocated.select_checkpoint_manager(
        CheckpointEngineConfig(backend="naive"), world.actor_wg, world.replicas
    )
    assert type(stock) is CheckpointEngineManager
    platform.device_name = "tpu"
    nccl = [_FakeReplica(world.events, 0, backend="nccl")]
    stock = tpu_colocated.select_checkpoint_manager(CheckpointEngineConfig(backend="naive"), world.actor_wg, nccl)
    assert type(stock) is CheckpointEngineManager


def test_install_manager_selector_binds_once_and_only_the_stock_name(monkeypatch):
    name = tpu_colocated.TRAINER_BASE_MODULE
    assert tpu_colocated.install_manager_selector() is False or name in sys.modules  # not imported here

    fake = ModuleType(name)
    fake.CheckpointEngineManager = CheckpointEngineManager
    monkeypatch.setitem(sys.modules, name, fake)
    assert tpu_colocated.install_manager_selector() is True
    assert fake.CheckpointEngineManager is tpu_colocated.select_checkpoint_manager
    assert tpu_colocated.install_manager_selector() is True  # idempotent

    fake.CheckpointEngineManager = object  # somebody else's binding is left alone
    assert tpu_colocated.install_manager_selector() is False
    assert fake.CheckpointEngineManager is object


def test_max_numeric_and_configured_backend():
    nested = [[{"a": 1, "b": 2.5, "flag": True}], {"a": 3, "s": "x"}]
    assert tpu_colocated._max_numeric(nested) == {"a": 3, "b": 2.5}
    assert tpu_colocated.configured_backend([]) is None
    assert tpu_colocated.configured_backend([SimpleNamespace(config=None)]) is None
