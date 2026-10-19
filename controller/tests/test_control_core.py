import json
import unittest
import urllib.request

from fakes import prestarted

from rl_pool_autoscaler import prom
from rl_pool_autoscaler.control_actor import ControlCore
from rl_pool_autoscaler.http_api import ControlHTTPServer
from rl_pool_autoscaler.orchestrator import PoolLimits, ScaleOrchestrator


def pool_status_from(b):
    def fn():
        return {
            "active": [{"id": r, "address": f"10.0.0.{i}:8000", "inflight": 3 * i} for i, r in enumerate(b.active)],
            "parked": [{"id": r, "address": ""} for r in b.parked],
            "max_concurrent_samples": 16 * len(b.active),
            "max_queue_size": 160,
            "stats": {"monitor/queue/mq_queue_size": 40, "monitor/active_tasks_size": 12},
            "lb_total_inflight": 12,
        }

    return fn


class ControlCoreHTTP(unittest.TestCase):
    def setUp(self):
        self.b = prestarted(4, 1)
        self.orch = ScaleOrchestrator(self.b, PoolLimits(1, 4), log=None)
        self.core = ControlCore(self.orch, pool_status_from(self.b), status_cache_s=0.0)
        self.srv = ControlHTTPServer(self.core, host="127.0.0.1", port=0).start()
        self.base = f"http://127.0.0.1:{self.srv.port}"

    def tearDown(self):
        self.srv.stop()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as r:
            return r.status, r.read().decode()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def test_metrics_exposes_required_series(self):
        self.core.push_trainer_metrics(
            {
                "current_param_version": 7,
                "trainer_idle_ratio": 0.62,
                "trainer_wait_s": 310.0,
                "trainer_step_s": 500.0,
                "last_sync_wall_s": 4.2,
                "mq_size": 40,
                "membership": {"versions": {"standalone_0": 7}},
            }
        )
        status, text = self.get("/metrics")
        self.assertEqual(status, 200)
        s = prom.parse(text)
        for name in [
            "rl_pool_active_standalone_replicas",
            "rl_pool_sleeping_standalone_replicas",
            "rl_pool_trainer_idle_ratio",
            "rl_pool_trainer_wait_s",
            "rl_pool_mq_size",
            "rl_pool_max_queue_size",
            "rl_pool_current_param_version",
            "rl_pool_last_sync_wall_s",
            "rl_pool_replica_inflight_requests",
        ]:
            self.assertIsNotNone(prom.scalar(s, name), name)
        self.assertEqual(prom.scalar(s, "rl_pool_active_standalone_replicas"), 1)
        self.assertEqual(prom.scalar(s, "rl_pool_sleeping_standalone_replicas"), 3)
        self.assertEqual(prom.scalar(s, "rl_pool_max_queue_size"), 160)
        self.assertEqual(prom.scalar(s, "rl_pool_current_param_version"), 7)
        self.assertEqual(prom.by_label(s, "rl_pool_replica_inflight_requests", "replica"), {"standalone_0": 0})
        self.assertEqual(prom.by_label(s, "rl_pool_replica_param_version", "replica"), {"standalone_0": 7})

    def test_scale_is_idempotent_and_clamped_over_http(self):
        st, r1 = self.post("/scale", {"standalone": 3})
        self.assertEqual(st, 200)
        self.assertEqual(r1["desired"], 3)
        self.assertFalse(r1["noop"])
        # reconcile synchronously (the real actor does this on its reconciler thread)
        while self.core.reconcile_once():
            pass
        self.assertEqual(len(self.b.active), 3)
        st, r2 = self.post("/scale", {"standalone": 3})
        self.assertTrue(r2["noop"])
        n = len(self.b.calls)
        self.assertIsNone(self.core.reconcile_once())
        self.assertEqual(len(self.b.calls), n)
        st, r3 = self.post("/scale", {"standalone": 99})
        self.assertEqual(r3["desired"], 4)
        self.assertTrue(r3["clamped"])
        st, r4 = self.post("/scale", {"standalone": 0})
        self.assertEqual(r4["desired"], 1)

    def test_bad_requests(self):
        st, body = self.post("/scale", {})
        self.assertEqual(st, 400)
        st, _ = self.get("/healthz")
        self.assertEqual(st, 200)
        with self.assertRaises(urllib.error.HTTPError):
            self.get("/nope")

    def test_status_json_and_in_progress_flag(self):
        self.post("/scale", {"standalone": 2})
        st, text = self.get("/status")
        js = json.loads(text)
        self.assertEqual(js["orchestrator"]["desired"], 2)
        self.assertEqual(js["orchestrator"]["active"], 1)
        s = prom.parse(self.get("/metrics")[1])
        self.assertEqual(prom.scalar(s, "rl_pool_scaling_in_progress"), 1)
        while self.core.reconcile_once():
            pass
        s = prom.parse(self.get("/metrics")[1])
        self.assertEqual(prom.scalar(s, "rl_pool_scaling_in_progress"), 0)

    def test_pool_status_error_is_cached_not_fatal(self):
        def boom():
            raise TimeoutError("rollouter busy")

        core = ControlCore(self.orch, boom, status_cache_s=0.0)
        text = core.metrics_text()
        self.assertIn("rl_pool_active_standalone_replicas 0", text)
        self.assertIn("rollouter busy", core.status()["pool_status_error"])


if __name__ == "__main__":
    unittest.main()
