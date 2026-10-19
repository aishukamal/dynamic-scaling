# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the TPU vLLM rollout glue.

vLLM, vllm-torchtpu and torch_tpu are not installed on CI, so ``vllm_async_server`` is replaced
by a minimal stand-in and every Ray actor handle is faked. The tests pin the contract with
upstream verl (which hooks are overridden and what they return) and the launch arguments.
"""

import asyncio
import importlib
import multiprocessing.process
import os
import pickle
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import ray
import ray.util.scheduling_strategies  # noqa: F401  (tpu_vllm reaches it through ``ray.util``)

# Import before any test patches sys.modules, so mock.patch.dict never evicts it.
from verl_hardware_plugin.accelerators.tpu.rollout import tpu_vllm_patches

VAS = "verl.workers.rollout.vllm_rollout.vllm_async_server"


class _FakeHttpServer:
    """Stands in for upstream ``vLLMHttpServer``: records the hooks the TPU subclass chains to."""

    def __init__(self):
        self.calls = []

    def _preprocess_engine_kwargs(self, engine_kwargs):
        self.calls.append("preprocess")
        engine_kwargs.setdefault("from_upstream", True)

    async def run_server(self, args):
        self.calls.append(("run_server", args))
        return "ran"

    def _get_worker_extension_cls(self):
        return "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"

    def _resolve_sleep_level(self):
        return "upstream"


class _FakeReplica:
    """Stands in for upstream ``vLLMReplica``."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.server_class = None
        self.init_hybrid_calls = []

    def _get_server_name_prefix(self):
        return "vllm_"

    async def init_hybrid(self, worker_group):
        self.init_hybrid_calls.append(worker_group)


@pytest.fixture
def tpu_vllm():
    """Import ``tpu_vllm`` against a stubbed ``vllm_async_server``, with ``patch_vllm_for_tpu`` mocked."""
    vas = ModuleType(VAS)
    vas.vLLMHttpServer = _FakeHttpServer
    vas.vLLMReplica = _FakeReplica
    with (
        mock.patch.dict(sys.modules, {VAS: vas}),
        mock.patch.object(tpu_vllm_patches, "patch_vllm_for_tpu"),
    ):
        sys.modules.pop("verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm", None)
        module = importlib.import_module("verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm")
        yield module
        sys.modules.pop("verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm", None)


def _done(value):
    future = asyncio.get_event_loop().create_future()
    future.set_result(value)
    return future


# ---------------------------------------------------------------------------
# Replica loader registration
# ---------------------------------------------------------------------------


def test_vllm_loader_defers_off_tpu_and_picks_tpu_replica_on_tpu():
    from verl.workers.rollout.replica import RolloutReplicaRegistry
    from verl_hardware_plugin.accelerators.tpu.registration import register_rollout

    sentinel_upstream = type("UpstreamReplica", (), {})
    sentinel_tpu = type("TPUvLLMReplica", (), {})
    fake_tpu_module = ModuleType("verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm")
    fake_tpu_module.TPUvLLMReplica = sentinel_tpu

    with mock.patch.dict(RolloutReplicaRegistry._registry, {"vllm": lambda: sentinel_upstream}):
        register_rollout()

        with mock.patch("verl.utils.device.get_resource_name", return_value="GPU"):
            assert RolloutReplicaRegistry.get("vllm") is sentinel_upstream

        with (
            mock.patch("verl.utils.device.get_resource_name", return_value="TPU"),
            mock.patch.dict(sys.modules, {"verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm": fake_tpu_module}),
        ):
            assert RolloutReplicaRegistry.get("vllm") is sentinel_tpu


# ---------------------------------------------------------------------------
# TPUvLLMHttpServer / TPUvLLMReplica hooks
# ---------------------------------------------------------------------------


def test_server_patches_vllm_before_upstream_init(tpu_vllm):
    assert not tpu_vllm.patch_vllm_for_tpu.called  # importing the module patches nothing
    order = []
    tpu_vllm.patch_vllm_for_tpu.side_effect = lambda: order.append("patch")
    with mock.patch.object(_FakeHttpServer, "__init__", lambda self: order.append("upstream init")):
        tpu_vllm.TPUvLLMHttpServer()
    assert order == ["patch", "upstream init"]


def test_server_engine_kwargs_force_tpu_executor(tpu_vllm):
    server = tpu_vllm.TPUvLLMHttpServer()
    engine_kwargs = {"distributed_executor_backend": "mp", "enable_sleep_mode": True}
    server._preprocess_engine_kwargs(engine_kwargs)

    assert server.calls == ["preprocess"]  # upstream validation still runs first
    assert engine_kwargs == {
        "distributed_executor_backend": "external_launcher",
        "enable_sleep_mode": True,  # upstream's setting is kept; the hybrid loop sleeps the worker
        "from_upstream": True,
    }


def test_server_sleep_level_is_2_for_tpu_weight_syncs(tpu_vllm):
    server = tpu_vllm.TPUvLLMHttpServer()
    assert server._resolve_sleep_level() == "upstream"  # no rollout config: upstream decides
    for backend in ("raiden", "tpu"):
        server.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(backend=backend))
        assert server._resolve_sleep_level() == 2
    server.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="naive"))
    assert server._resolve_sleep_level() == "upstream"


def test_server_worker_extension_defers_to_upstream(tpu_vllm):
    upstream_cls = "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"
    server = tpu_vllm.TPUvLLMHttpServer()
    assert server._get_worker_extension_cls() == upstream_cls  # no rollout config
    server.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="tpu"))
    assert server._get_worker_extension_cls() == upstream_cls
    # No run_server override: vLLM v0.29 has no V0 engine to switch off.
    assert "run_server" not in tpu_vllm.TPUvLLMHttpServer.__dict__


def test_server_worker_extension_for_raiden(tpu_vllm):
    server = tpu_vllm.TPUvLLMHttpServer()
    server.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="raiden"))
    assert server._get_worker_extension_cls() == tpu_vllm.RAIDEN_WORKER_EXTENSION_CLS
    module_name, cls_name = tpu_vllm.RAIDEN_WORKER_EXTENSION_CLS.rsplit(".", 1)
    assert (module_name, cls_name) == (
        "verl_hardware_plugin.accelerators.tpu.rollout.tpu_raiden",
        "vLLMRaidenWorkerExtension",
    )


def test_replica_uses_tpu_server(tpu_vllm):
    replica = tpu_vllm.TPUvLLMReplica(0, "config", "model_config")
    assert replica.server_class is not None
    assert replica.args == (0, "config", "model_config")


# ---------------------------------------------------------------------------
# launch_tpu_vllm_servers
# ---------------------------------------------------------------------------


class _FakeRemoteMethod:
    def __init__(self, result):
        self._result = result

    def remote(self, *args, **kwargs):
        return _done(self._result(*args, **kwargs) if callable(self._result) else self._result)


class _FakeWorker:
    def __init__(self, node_id, chip="0", ip=None, physical_host=None):
        self._node_id = node_id
        self._chip = chip
        self._ip = ip or f"10.0.0.{(sum(map(ord, node_id[:4])) % 200) + 1}"
        self._physical_host = physical_host
        self.calls = []
        self.__ray_call__ = _FakeRemoteMethod(self._call)

    def _call(self, fn):
        # launch_tpu_vllm_servers asks the workers for their node ids (and chips for hybrid replicas);
        # init_hybrid also asks for their node IP and physical host.
        self.calls.append(fn)
        env = {"TPU_VISIBLE_CHIPS": self._chip, "VERL_TPU_PHYSICAL_HOST": self._physical_host or ""}
        with (
            mock.patch("ray.get_runtime_context", return_value=SimpleNamespace(get_node_id=lambda: self._node_id)),
            mock.patch("ray.util.get_node_ip_address", return_value=self._ip),
            mock.patch.dict(os.environ, env),
        ):
            return fn(self)


class _FakeServer:
    get_master_address = _FakeRemoteMethod(("10.0.0.1", 1234, 5678))
    get_server_address = _FakeRemoteMethod(("10.0.0.1", 8000))

    def __init__(self):
        self.launch_kwargs = None
        self.launch_server = _FakeRemoteMethod(self._launch)

    def _launch(self, **kwargs):
        self.launch_kwargs = kwargs


class _FakeServerClass:
    def __init__(self):
        self.options_kwargs = None
        self.init_kwargs = None
        self.server = _FakeServer()

    def options(self, **kwargs):
        self.options_kwargs = kwargs
        return self

    def remote(self, **kwargs):
        self.init_kwargs = kwargs
        return self.server


NODE_A = "a" * 56
NODE_B = "b" * 56


def _fake_replica(data_parallel_size=1, rollout_mode="standalone"):
    workers = [_FakeWorker(node, chip=str(chip)) for node in (NODE_A, NODE_B) for chip in range(4)]
    return SimpleNamespace(
        config=SimpleNamespace(
            data_parallel_size=data_parallel_size, tensor_model_parallel_size=8, ray_actor_max_concurrency=1234
        ),
        model_config="model_config",
        rollout_mode=rollout_mode,
        workers=workers,
        replica_rank=3,
        name_suffix="",
        is_reward_model=False,
        is_teacher_model=False,
        gpus_per_replica_node=4,
        nnodes=2,
        resource_pool=SimpleNamespace(
            get_placement_groups=lambda device_name=None: [
                SimpleNamespace(id=SimpleNamespace(hex=lambda: "pg0hex")),
                SimpleNamespace(id=SimpleNamespace(hex=lambda: "pg1hex")),
            ]
        ),
        servers=[],
        server_class=_FakeServerClass(),
        _get_server_name_prefix=lambda: "vllm_",
    )


def test_launch_tpu_vllm_servers_single_server_spanning_all_workers(tpu_vllm):
    replica = _fake_replica()
    platform = SimpleNamespace(ray_noset_envvars=lambda: ["RAY_NOSET"], rollout_env_vars=lambda: {})

    async def run():
        with mock.patch.object(tpu_vllm, "get_platform", return_value=platform):
            await tpu_vllm.launch_tpu_vllm_servers(replica)

    asyncio.run(run())

    server_class = replica.server_class
    options = server_class.options_kwargs
    assert options["name"] == "vllm_server_3_0"
    assert options["max_concurrency"] == 1234
    assert options["scheduling_strategy"].node_id == NODE_A
    assert options["runtime_env"] == {
        "env_vars": {
            "RAY_NOSET": "1",
            "TPU_MULTIHOST_BACKEND": "ray",
            "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "0",
            "VERL_TPU_PG_IDS": "pg0hex,pg1hex",
        }
    }

    init = server_class.init_kwargs
    assert init["workers"] is replica.workers
    assert init["node_rank"] == 0
    assert init["nnodes"] == 2
    assert init["cuda_visible_devices"] == ""

    assert server_class.server.launch_kwargs == {"master_address": "10.0.0.1", "master_port": 1234, "dp_rpc_port": 5678}
    assert replica.servers == [server_class.server]
    assert replica._server_address == "10.0.0.1:8000"


def test_launch_tpu_vllm_servers_hybrid_places_by_trainer_workers(tpu_vllm):
    from verl.workers.rollout.replica import RolloutMode

    replica = _fake_replica(rollout_mode=RolloutMode.HYBRID)
    platform = SimpleNamespace(ray_noset_envvars=lambda: ["RAY_NOSET"], rollout_env_vars=lambda: {})

    async def run():
        with mock.patch.object(tpu_vllm, "get_platform", return_value=platform):
            await tpu_vllm.launch_tpu_vllm_servers(replica)

    asyncio.run(run())

    options = replica.server_class.options_kwargs
    assert options["scheduling_strategy"].node_id == NODE_A
    env = options["runtime_env"]["env_vars"]
    assert "VERL_TPU_PG_IDS" not in env  # no placement group, no resource pool
    expected = ",".join(f"{node}:{chip}" for node in (NODE_A, NODE_B) for chip in range(4))
    assert env["VERL_TPU_HYBRID_WORKERS"] == expected
    assert env["TPU_MULTIHOST_BACKEND"] == "ray"
    # Every trainer worker was asked for its placement, none was suspended by the launch itself.
    assert all(worker.calls == [tpu_vllm._trainer_worker_placement] for worker in replica.workers)
    assert replica.server_class.init_kwargs["cuda_visible_devices"] == ""


def test_replica_init_hybrid_suspends_its_trainer_workers_then_launches(tpu_vllm):
    from verl_hardware_plugin.accelerators.tpu import platform_tpu

    suspended = []
    workers = [_FakeWorker(NODE_A, chip=str(i)) for i in range(8)]
    worker_group = SimpleNamespace(workers=workers)
    replica = tpu_vllm.TPUvLLMReplica(1, "config", "model_config")
    replica.world_size, replica.replica_rank = 4, 1

    with mock.patch.object(platform_tpu, "suspend_device", side_effect=lambda: suspended.append(1) or {"ok": 1}):
        asyncio.run(replica.init_hybrid(worker_group))

    # Replica 1 of world size 4 owns workers 4..7: each was suspended once, the others only asked for their host.
    assert [worker.calls for worker in workers[:4]] == [[tpu_vllm._trainer_worker_host]] * 4
    assert [worker.calls for worker in workers[4:]] == [[tpu_vllm._trainer_worker_host, tpu_vllm._suspend_trainer_worker]] * 4
    assert len(suspended) == 4
    assert replica.init_hybrid_calls == [worker_group]  # upstream's init_hybrid runs after the suspend


# Four hosts (VMs) of 4 chips; two VMs of one physical machine report the same ``physical_host`` value, here
# (A, C) and (B, D), while IP order (and worker order) is A, B, C, D. A TP=8 replica must land on one machine.
NODE_C, NODE_D = "c" * 56, "d" * 56
PAIRS = {NODE_A: "/c1/sb0/m7", NODE_C: "/c1/sb0/m7", NODE_B: "/c1/sb0/m9", NODE_D: "/c1/sb0/m9"}
IPS = {NODE_A: "10.0.0.1", NODE_B: "10.0.0.2", NODE_C: "10.0.0.3", NODE_D: "10.0.0.4"}


def _four_hosts(physical=True):
    return [
        _FakeWorker(node, chip=str(chip), ip=IPS[node], physical_host=PAIRS[node] if physical else None)
        for node in (NODE_A, NODE_B, NODE_C, NODE_D)
        for chip in range(4)
    ]


def _hosts_of(workers):
    return [w._call(lambda self: (self._node_id, self._ip, os.environ.get("VERL_TPU_PHYSICAL_HOST") or None)) for w in workers]


def test_hybrid_replica_workers_groups_multi_host_replicas_by_physical_host(tpu_vllm, monkeypatch):
    monkeypatch.delenv("VERL_TPU_HOST_ORDER", raising=False)
    workers = _four_hosts()
    hosts = _hosts_of(workers)
    r0, how0 = tpu_vllm.hybrid_replica_workers(workers, 8, 0, hosts)
    r1, how1 = tpu_vllm.hybrid_replica_workers(workers, 8, 1, hosts)
    assert (how0, how1) == ("physical_host", "physical_host")
    assert [w._node_id[0] for w in r0] == ["a"] * 4 + ["c"] * 4 and [w._chip for w in r0] == ["0", "1", "2", "3"] * 2
    assert [w._node_id[0] for w in r1] == ["b"] * 4 + ["d"] * 4
    # single-host replicas keep upstream's rank-order slice
    r2, how2 = tpu_vllm.hybrid_replica_workers(workers, 4, 2, hosts)
    assert how2 == "rank order" and r2 == workers[8:12]
    # VERL_TPU_HOST_ORDER (host IPs in physical order) wins over the metadata
    monkeypatch.setenv("VERL_TPU_HOST_ORDER", "10.0.0.4,10.0.0.1,10.0.0.3,10.0.0.2")
    r0, how0 = tpu_vllm.hybrid_replica_workers(workers, 8, 0, hosts)
    assert how0 == "VERL_TPU_HOST_ORDER" and [w._node_id[0] for w in r0] == ["d"] * 4 + ["a"] * 4
    # neither available: rank order, with a warning
    monkeypatch.delenv("VERL_TPU_HOST_ORDER")
    workers_plain = _four_hosts(physical=False)
    r0, how0 = tpu_vllm.hybrid_replica_workers(workers_plain, 8, 0, _hosts_of(workers_plain))
    assert how0 == "rank order" and r0 == workers_plain[:8]
    with pytest.raises(ValueError):
        tpu_vllm.hybrid_replica_workers(workers, 6, 0, hosts)


def test_replica_init_hybrid_multi_host_uses_machine_pairs(tpu_vllm, monkeypatch):
    from verl_hardware_plugin.accelerators.tpu import platform_tpu

    monkeypatch.delenv("VERL_TPU_HOST_ORDER", raising=False)
    workers = _four_hosts()
    worker_group = SimpleNamespace(workers=workers)
    replica = tpu_vllm.TPUvLLMReplica(1, "config", "model_config")
    replica.world_size, replica.replica_rank = 8, 1
    launched = []
    replica.launch_servers = _FakeRemoteMethod(lambda: launched.append(1))._result  # plain coroutine stand-in

    async def fake_launch():
        launched.append(1)

    replica.launch_servers = fake_launch
    with mock.patch.object(platform_tpu, "suspend_device", return_value={"ok": 1}):
        asyncio.run(replica.init_hybrid(worker_group))
    # replica 1 = machine pair (B, D): those 8 workers were suspended, the others only probed
    assert [w._node_id[0] for w in replica.workers] == ["b"] * 4 + ["d"] * 4
    assert replica.rollout_mode == tpu_vllm.RolloutMode.HYBRID and launched == [1]
    assert replica.init_hybrid_calls == []  # upstream's rank-order slice was not used
    suspended = [w for w in workers if tpu_vllm._suspend_trainer_worker in w.calls]
    assert sorted({w._node_id[0] for w in suspended}) == ["b", "d"] and len(suspended) == 8


def test_server_collective_rpc_returns_worker_results(tpu_vllm):
    server = tpu_vllm.TPUvLLMHttpServer()

    async def engine_rpc(method, timeout=None, args=(), kwargs=None):
        return [f"{method}:{args}:{kwargs}:{timeout}"]

    server.engine = SimpleNamespace(collective_rpc=engine_rpc)
    result = asyncio.run(server.collective_rpc("install_raiden_weights", timeout=3, args=(1,), kwargs={"a": 2}))
    assert result == ["install_raiden_weights:(1,):{'a': 2}:3"]


def test_launch_tpu_vllm_servers_rejects_data_parallel(tpu_vllm):
    with pytest.raises(NotImplementedError, match="data_parallel_size"):
        asyncio.run(tpu_vllm.launch_tpu_vllm_servers(_fake_replica(data_parallel_size=2)))


# ---------------------------------------------------------------------------
# patch_vllm_for_tpu (without vllm-torchtpu installed)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_patches():
    """Run patch_vllm_for_tpu with its process-global side effects rolled back afterwards."""

    base_process = multiprocessing.process.BaseProcess
    saved_init = base_process.__init__
    had_flag = "_tpu_patched" in base_process.__dict__
    env = {k: v for k, v in os.environ.items() if k != "VLLM_DISABLE_COMPILE_CACHE"}
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(tpu_vllm_patches, "_PATCHES_APPLIED", False),
    ):
        yield tpu_vllm_patches
    base_process.__init__ = saved_init
    if not had_flag and "_tpu_patched" in base_process.__dict__:
        del base_process._tpu_patched


def test_patch_vllm_for_tpu_env_side_effects_without_torchtpu(isolated_patches):
    isolated_patches.patch_vllm_for_tpu()

    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"
    assert multiprocessing.process.BaseProcess._tpu_patched is True
    # The executor patches need vllm-torchtpu, so the guard stays open for a later retry.
    assert isolated_patches._PATCHES_APPLIED is False


def test_patch_vllm_for_tpu_is_a_no_op_once_applied(isolated_patches):
    isolated_patches._PATCHES_APPLIED = True
    # vLLM is made unimportable, so getting past the guard would log "Skipping vLLM TPU executor patches".
    with mock.patch.dict(sys.modules, {"vllm": None}), mock.patch.object(isolated_patches.logger, "debug") as debug:
        isolated_patches.patch_vllm_for_tpu()
    debug.assert_not_called()
    assert "VLLM_DISABLE_COMPILE_CACHE" not in os.environ


def test_patch_vllm_for_tpu_keeps_explicit_compile_cache_setting(isolated_patches):
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "0"
    isolated_patches.patch_vllm_for_tpu()
    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "0"


def test_pickleable_process_wrapper_applies_patches_before_target():
    wrapper = pickle.loads(pickle.dumps(tpu_vllm_patches.PickleableProcessWrapper(len)))
    with mock.patch.object(tpu_vllm_patches, "patch_vllm_for_tpu") as patch:
        assert wrapper([1, 2, 3]) == 3
    patch.assert_called_once_with()


def test_multiprocessing_patch_wraps_keyword_and_positional_targets(isolated_patches):
    isolated_patches.patch_multiprocessing_for_tpu()
    for process in (multiprocessing.Process(target=len), multiprocessing.Process(None, len)):
        assert isinstance(process._target, isolated_patches.PickleableProcessWrapper)
        assert process._target.target is len
    assert multiprocessing.Process()._target is None


def test_local_ranks_count_workers_per_host_in_rank_order():
    assert tpu_vllm_patches._local_ranks(["b", "b", "a", "a", "a"]) == [0, 1, 0, 1, 2]


def test_hybrid_worker_specs_parse_and_validate():
    assert tpu_vllm_patches._hybrid_worker_specs("") == []
    assert tpu_vllm_patches._hybrid_worker_specs(f"{NODE_A}:2,{NODE_B}:0") == [(NODE_A, 2), (NODE_B, 0)]
    with pytest.raises(ValueError, match="VERL_TPU_HYBRID_WORKERS"):
        tpu_vllm_patches._hybrid_worker_specs("nodeonly")


def test_tpu_worker_envs_hybrid_use_the_trainer_chips():
    ips = ["10.0.0.2"] * 2 + ["10.0.0.1"] * 2
    env = {k: v for k, v in os.environ.items() if k != "TORCH_TPU_TOPOLOGY"}
    with mock.patch.dict(os.environ, env, clear=True):
        envs = tpu_vllm_patches._tpu_worker_envs(ips, base_port=8070, chips=[2, 3, 2, 3])
    # Two TP=2 replicas can share a host: this one drives chips 2 and 3 and serves the slice builder on
    # 8072 / 8073, disjoint from a sibling replica on chips 0 / 1 and from the trainer's 8471+ range.
    assert [e["TPU_VISIBLE_CHIPS"] for e in envs] == ["2", "3", "2", "3"]
    assert [e["TPU_VISIBLE_DEVICES"] for e in envs] == ["2", "3", "2", "3"]
    assert [e["TPU_PROCESS_PORT"] for e in envs] == ["8072", "8073", "8072", "8073"]
    assert envs[0]["TPU_PROCESS_ADDRESSES"] == "10.0.0.2:8072,10.0.0.2:8073,10.0.0.1:8072,10.0.0.1:8073"
    assert envs[0]["TORCH_TPU_TOPOLOGY"] == "2,2,1"


def test_tpu_worker_envs_two_hosts_of_four_chips():
    ips = ["10.0.0.2"] * 4 + ["10.0.0.1"] * 4  # rank order: the driver's host first
    env = {k: v for k, v in os.environ.items() if k != "TORCH_TPU_TOPOLOGY"}
    with mock.patch.dict(os.environ, env, clear=True):
        envs = tpu_vllm_patches._tpu_worker_envs(ips, base_port=8070)

    addresses = ",".join(f"{ip}:{8070 + chip}" for ip in ("10.0.0.2", "10.0.0.1") for chip in range(4))
    assert envs[5] == {
        "TPU_VISIBLE_CHIPS": "1",
        "TPU_VISIBLE_DEVICES": "1",
        "TPU_PROCESS_PORT": "8071",
        "TPU_PROCESS_ADDRESSES": addresses,
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": addresses,
        "CLOUD_TPU_TASK_ID": "1",
        "TPU_WORKER_HOSTNAMES": "10.0.0.2,10.0.0.1",
        # One process per chip: libtpu sees 8 single-chip "hosts" laid out as the 2x4 slice.
        "TORCH_TPU_TOPOLOGY": "2,4,1",
        "TPU_HOST_BOUNDS": "2,4,1",
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
    }
    assert [e["TPU_VISIBLE_CHIPS"] for e in envs] == ["0", "1", "2", "3"] * 2
    assert [e["CLOUD_TPU_TASK_ID"] for e in envs] == ["0"] * 4 + ["1"] * 4
