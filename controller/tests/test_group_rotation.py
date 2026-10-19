import unittest
from unittest import mock


class FakeWG:
    def __init__(self, ws):
        self.world_size = ws
        self.calls = []

    def execute_checkpoint_engine(self, *args):
        self.calls.append(tuple(a[0] if isinstance(a, list) else a for a in args))
        return ["ref"] * self.world_size


class GroupRotation(unittest.TestCase):
    def setUp(self):
        import sys, types
        # trainer.py imports verl/ray at module level; stub what the rotation helper needs
        self._mods = {}
        for name in ("ray", "verl", "verl.checkpoint_engine", "verl.experimental", "verl.experimental.fully_async_policy",
                     "verl.experimental.fully_async_policy.fully_async_trainer", "verl.utils", "verl.utils.config"):
            self._mods[name] = sys.modules.get(name)
        ray = types.ModuleType("ray"); ray.get = lambda refs, **kw: refs; sys.modules["ray"] = ray
        ce = types.ModuleType("verl.checkpoint_engine"); ce.CheckpointEngineManager = type("CheckpointEngineManager", (), {"build_process_group": lambda self, r: "built"})
        sys.modules["verl"] = types.ModuleType("verl"); sys.modules["verl.checkpoint_engine"] = ce
        ft = types.ModuleType("verl.experimental.fully_async_policy.fully_async_trainer"); ft.FullyAsyncTrainer = type("FullyAsyncTrainer", (), {})
        sys.modules["verl.experimental"] = types.ModuleType("verl.experimental"); sys.modules["verl.experimental.fully_async_policy"] = types.ModuleType("verl.experimental.fully_async_policy"); sys.modules["verl.experimental.fully_async_policy.fully_async_trainer"] = ft
        uc = types.ModuleType("verl.utils.config"); uc.omega_conf_to_dataclass = lambda x: x
        sys.modules["verl.utils"] = types.ModuleType("verl.utils"); sys.modules["verl.utils.config"] = uc
        sys.modules.pop("rl_pool_autoscaler.trainer", None)
        import rl_pool_autoscaler.trainer as tr
        self.tr = tr
        tr._GROUP_STATE["prev"] = None
        tr._GROUP_STATE["events"].clear()

    def tearDown(self):
        import sys
        sys.modules.pop("rl_pool_autoscaler.trainer", None)
        for k, v in self._mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_first_sync_only_renames(self):
        a, r = FakeWG(8), FakeWG(1)
        name = self.tr.rotate_collective_group(a, r, destroy_prev=True)
        self.assertTrue(name.startswith("rlpool_"))
        self.assertEqual(a.calls, [("__setattr__", "group_name", name)])
        self.assertEqual(r.calls, [("__setattr__", "group_name", name)])

    def test_second_sync_destroys_previous_then_renames(self):
        a, r1 = FakeWG(8), FakeWG(1)
        n1 = self.tr.rotate_collective_group(a, r1, destroy_prev=True)
        a.calls.clear()
        r2 = FakeWG(2)  # membership changed: 2 replicas now
        n2 = self.tr.rotate_collective_group(a, r2, destroy_prev=True)
        self.assertNotEqual(n1, n2)
        self.assertEqual(a.calls, [("__setattr__", "group_name", n1), ("__setattr__", "rebuild_group", True), ("__setattr__", "rank", 0), ("finalize",), ("__setattr__", "rebuild_group", False), ("__setattr__", "group_name", n2)])
        self.assertEqual(r2.calls, a.calls)
        self.assertEqual(self.tr._GROUP_STATE["events"][-1]["destroyed"], n1)

    def test_no_destroy_mode(self):
        a, r = FakeWG(8), FakeWG(1)
        self.tr.rotate_collective_group(a, r, destroy_prev=False)
        a.calls.clear()
        n2 = self.tr.rotate_collective_group(a, r, destroy_prev=False)
        self.assertEqual(a.calls, [("__setattr__", "group_name", n2)])


if __name__ == "__main__":
    unittest.main()
