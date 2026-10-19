"""Launcher: drop-in for ``verl.experimental.fully_async_policy.fully_async_main``.

    python -m rl_pool_autoscaler.launcher <hydra overrides...>

Builds the STOCK components exactly like ``FullyAsyncTaskRunner._initialize_components`` (the base
method is reused unchanged), substituting:
- ``ElasticRollouter`` for ``FullyAsyncRollouter`` (``_create_rollouter``), plus the prestart park
  step between ``init_workers()`` and ``set_max_required_samples()``;
- ``ElasticTrainer`` for ``FullyAsyncTrainer`` (``_create_trainer``);
and then creates the named control actor on the Ray head node.

The trainer registry (``async_training.trainer_name``) does not exist at verl d4701e4e, so the
subclass is selected here instead of by config; if a future image carries the registry the same
``ElasticTrainer`` class can be registered under ``rl_pool`` without changes.
"""

from __future__ import annotations

import os
import sys

import hydra
import ray

from verl.experimental.fully_async_policy import fully_async_main as _stock_main
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner as _StockRunner
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device

from rl_pool_autoscaler._compat import head_node_id, rl_pool_config, unwrap_actor_class
from rl_pool_autoscaler.control_actor import RLPoolControl
from rl_pool_autoscaler.rollouter import ElasticRollouter
from rl_pool_autoscaler.trainer import ElasticTrainer

_BaseRunner = unwrap_actor_class(_StockRunner)


class RLPoolTaskRunner(_BaseRunner):
    def _create_rollouter(self, config) -> None:
        pool = rl_pool_config(config)
        print(f"[RL POOL MAIN] Starting create ElasticRollouter... rl_pool={pool}")
        rollouter = ray.remote(num_cpus=10, max_concurrency=100)(ElasticRollouter).remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        if "hybrid_worker_group" in self.components:  # stock behaviour, kept for parity (hybrid is OFF in the PoC)
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
        ray.get(rollouter.init_workers.remote())
        n_initial = len(ray.get(rollouter.get_active_standalone_ids.remote()))
        if pool.prestart:
            info = ray.get(rollouter.prestart_park.remote(pool.min_standalone))
            print(f"[RL POOL MAIN] prestart: {n_initial} replicas created, parked {len(info['parked'])}, active {info['active']}")
            if n_initial < pool.max_standalone:
                print(
                    f"[RL POOL MAIN] WARNING prestart=true but only {n_initial} replicas fit rollout.nnodes x "
                    f"rollout.n_gpus_per_node; scale-outs beyond {n_initial} will cold-start new replicas"
                )
        elif n_initial != pool.min_standalone:
            print(f"[RL POOL MAIN] WARNING prestart=false: initial active {n_initial} != rl_pool.min_standalone {pool.min_standalone}")
        ray.get(rollouter.set_max_required_samples.remote())
        self.components["rollouter"] = rollouter
        print("[RL POOL MAIN] ElasticRollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        print("[RL POOL MAIN] Starting create ElasticTrainer...")
        trainer_role_mapping = {
            role: worker_cls for role, worker_cls in self.components["role_worker_mapping"].items() if role != Role.Rollout
        }
        trainer = ray.remote(num_cpus=10)(ElasticTrainer).remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[RL POOL MAIN] ElasticTrainer created and initialized successfully")

    def _initialize_components(self, config) -> None:
        super()._initialize_components(config)
        self._create_control_actor(config)

    def _create_control_actor(self, config) -> None:
        pool = rl_pool_config(config)
        options: dict = {"name": pool.control_actor_name, "num_cpus": 0, "max_concurrency": 4}
        head = head_node_id()
        if head:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            # hard affinity: with soft=True Ray placed the actor on the trainer node in the first live run
            # (m2-pods1b, 2026-10-15), which breaks the Service that selects the head pod for :8787.
            options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(node_id=head, soft=False)
        kwargs = dict(
            trainer=self.components["trainer"],
            rollouter=self.components["rollouter"],
            min_standalone=pool.min_standalone,
            max_standalone=pool.max_standalone,
            port=pool.control_port,
            release_on_scale_in=pool.release_on_scale_in,
            join_timeout_s=pool.join_timeout_s,
            join_poll_s=pool.join_poll_s,
            scale_timeout_s=pool.scale_timeout_s,
            status_timeout_s=pool.status_timeout_s,
        )
        ctl = ray.remote(RLPoolControl).options(**options).remote(**kwargs)
        try:
            info = ray.get(ctl.ready.remote(), timeout=180)
        except Exception as e:  # noqa: BLE001 - hard head affinity could not be satisfied: fall back to anywhere
            print(f"[RL POOL MAIN] control actor did not start on the head within 180 s ({type(e).__name__}); retrying without affinity")
            try:
                ray.kill(ctl)
            except Exception:  # noqa: BLE001
                pass
            options.pop("scheduling_strategy", None)
            ctl = ray.remote(RLPoolControl).options(**options).remote(**kwargs)
            info = ray.get(ctl.ready.remote())
        try:
            ctl_ip = ray.get(ctl.node_ip.remote(), timeout=30)
        except Exception:  # noqa: BLE001
            ctl_ip = None
        print(f"[RL POOL MAIN] control actor node ip={ctl_ip}")
        ray.get(self.components["trainer"].set_control_actor.remote(ctl))
        self.components["rl_pool_control"] = ctl
        print(f"[RL POOL MAIN] control actor '{pool.control_actor_name}' ready on head={head} port={info['port']}")

    def _run_training_loop(self):
        try:
            super()._run_training_loop()
        finally:
            ctl = self.components.get("rl_pool_control")
            if ctl is not None:
                try:
                    ray.get(ctl.shutdown.remote(), timeout=30)
                except Exception as e:  # noqa: BLE001
                    print(f"[RL POOL MAIN] control actor shutdown: {e}")
                try:
                    ray.kill(ctl)
                except Exception:  # noqa: BLE001
                    pass


# The stock fully_async_ppo_trainer.yaml sets hydra.searchpath, and Hydra only allows that in the
# PRIMARY config ("Overriding hydra.searchpath is only supported from the primary config", measured
# in the verl-swe-poc:v2 image), so it cannot be pulled in as a default of our own yaml. The stock
# config is therefore used as the primary config and the ``rl_pool.*`` overrides are turned into
# ``+rl_pool.*`` appends before Hydra parses argv (see ``_rewrite_argv``). config/rl_pool_fully_async.yaml
# stays as the documented list of keys and defaults (``RLPoolConfig`` holds the same defaults).
_STOCK_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(_stock_main.__file__)), "config")


def _rewrite_argv(argv: list[str]) -> list[str]:
    out = []
    for a in argv:
        if a.startswith("rl_pool.") and "=" in a:
            out.append("+" + a)
        elif a.startswith("++rl_pool."):
            out.append(a)
        else:
            out.append(a)
    return out


@hydra.main(config_path=_STOCK_CONFIG_DIR, config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from time import time

    from verl.trainer.main_ppo import run_ppo

    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")
    rl_pool_config(config)  # validate early
    if config.async_training.get("use_dynamic_resource_scheduling", False):
        raise RuntimeError("rl_pool: async_training.use_dynamic_resource_scheduling must be False (hybrid scheduling is off by design)")
    start_time = time()
    auto_set_device(config)
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(RLPoolTaskRunner))
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + _rewrite_argv(sys.argv[1:])
    main()
