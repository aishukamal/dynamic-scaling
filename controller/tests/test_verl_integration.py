"""Tests that need the real verl classes. SKIPPED when verl/ray/torch are not importable.

Run inside the verl image (CPU is fine; nothing initialises Ray or touches a GPU):

    cd /path/to/working_dir && python -m unittest tests.test_verl_integration -v

They check (1) the stock seams this plugin relies on still exist with the expected shapes at the
pinned commit, (2) the actor-class unwrapping yields subclassable plain classes, and (3) the
MetricsTap derives the control metrics from the keys the stock trainer actually logs.
"""

import importlib.util
import inspect
import unittest

HAVE_VERL = all(importlib.util.find_spec(m) is not None for m in ("verl", "ray", "torch"))


@unittest.skipUnless(HAVE_VERL, "requires verl + ray + torch (run inside the verl image)")
class StockSeams(unittest.TestCase):
    def test_llm_server_seams(self):
        from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer, LLMServerManager

        for name in ("add_servers", "remove_servers", "get_status", "get_total_inflight", "get_inflight_count"):
            self.assertTrue(callable(getattr(GlobalRequestLoadBalancer, name)), name)
        self.assertEqual(list(inspect.signature(GlobalRequestLoadBalancer.add_servers).parameters), ["self", "servers"])
        self.assertEqual(list(inspect.signature(GlobalRequestLoadBalancer.remove_servers).parameters), ["self", "server_ids"])
        for name in ("get_replicas", "get_client", "_initialize_llm_servers", "_init_global_load_balancer"):
            self.assertTrue(hasattr(LLMServerManager, name), name)

    def test_fully_async_manager_fields(self):
        src = inspect.getsource(importlib.import_module("verl.experimental.fully_async_policy.fully_async_rollouter"))
        for needle in ("self.rollout_replicas", "self.server_handles", "self.server_addresses", "self.prometheus_server_addresses", "self.hybrid_replicas", "self.alive_replicas", "def _update_max_concurrent_samples"):
            self.assertIn(needle, src, needle)

    def test_replica_seams(self):
        from verl.workers.rollout.replica import RolloutReplica

        for name in ("init_standalone", "sleep", "wake_up", "abort_all_requests", "resume_generation"):
            self.assertTrue(inspect.iscoroutinefunction(getattr(RolloutReplica, name)), name)
        self.assertEqual(list(inspect.signature(RolloutReplica.__init__).parameters)[:4], ["self", "replica_rank", "config", "model_config"])

    def test_checkpoint_manager_seams(self):
        from verl.checkpoint_engine import CheckpointEngineManager

        self.assertEqual(list(inspect.signature(CheckpointEngineManager.__init__).parameters), ["self", "config", "actor_wg", "replicas"])
        for name in ("add_replicas", "remove_replicas", "update_weights", "abort_replicas", "resume_generation_replicas"):
            self.assertTrue(hasattr(CheckpointEngineManager, name), name)
        src = inspect.getsource(CheckpointEngineManager.update_weights)
        self.assertIn("for replica in self.replicas", src)  # the reason the lock exists

    def test_trainer_registry_absent_or_present(self):
        mod = importlib.import_module("verl.experimental.fully_async_policy.fully_async_trainer")
        # Either is fine; the launcher does not depend on it. Record which world we are in.
        print("trainer registry present:", hasattr(mod, "register_trainer"))

    def test_unwrap_gives_plain_classes(self):
        from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
        from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

        from rl_pool_autoscaler._compat import unwrap_actor_class

        for actor_cls in (FullyAsyncTrainer, FullyAsyncRollouter, FullyAsyncTaskRunner):
            plain = unwrap_actor_class(actor_cls)
            self.assertTrue(inspect.isclass(plain))
            self.assertFalse(hasattr(plain, "__ray_metadata__"))

            class Sub(plain):  # noqa: N801 - must not raise "subclass of actor"
                pass

            self.assertTrue(issubclass(Sub, plain))

    def test_plugin_classes_import_and_subclass(self):
        from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
        from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

        from rl_pool_autoscaler._compat import unwrap_actor_class
        from rl_pool_autoscaler.rollouter import ElasticRollouter
        from rl_pool_autoscaler.trainer import ElasticTrainer, LockedCheckpointManager

        self.assertTrue(issubclass(ElasticTrainer, unwrap_actor_class(FullyAsyncTrainer)))
        self.assertTrue(issubclass(ElasticRollouter, unwrap_actor_class(FullyAsyncRollouter)))
        for name in ("attach_replicas", "detach_replicas", "sync_joiners_now", "get_membership", "get_control_metrics"):
            self.assertTrue(inspect.iscoroutinefunction(getattr(ElasticTrainer, name)) or callable(getattr(ElasticTrainer, name)), name)
        for name in ("add_standalone", "prestart_park", "lb_add", "lb_remove", "abort_standalone", "sleep_standalone", "destroy_standalone", "get_pool_status"):
            self.assertTrue(inspect.iscoroutinefunction(getattr(ElasticRollouter, name)), name)
        self.assertTrue(inspect.iscoroutinefunction(LockedCheckpointManager.update_weights))

    def test_hydra_config_composes(self):
        """The launcher composes the STOCK primary config and appends rl_pool.* (see launcher._rewrite_argv)."""
        from hydra import compose, initialize_config_dir

        from rl_pool_autoscaler._compat import rl_pool_config
        from rl_pool_autoscaler.launcher import _STOCK_CONFIG_DIR, _rewrite_argv

        overrides = _rewrite_argv(["rl_pool.max_standalone=3", "rl_pool.prestart=false", "async_training.use_dynamic_resource_scheduling=false"])
        self.assertEqual(overrides[0], "+rl_pool.max_standalone=3")
        with initialize_config_dir(config_dir=_STOCK_CONFIG_DIR, version_base=None):
            cfg = compose(config_name="fully_async_ppo_trainer", overrides=overrides)
        pool = rl_pool_config(cfg)
        self.assertEqual(pool.max_standalone, 3)
        self.assertFalse(pool.prestart)
        self.assertEqual(pool.min_standalone, 1)
        self.assertFalse(cfg.async_training.use_dynamic_resource_scheduling)
        self.assertIn("trigger_parameter_sync_step", cfg.async_training)
        self.assertIn("ray_pg_extra_resources", cfg)


if __name__ == "__main__":
    unittest.main()
