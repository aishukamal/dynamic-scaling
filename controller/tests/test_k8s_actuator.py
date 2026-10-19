import unittest

from rl_pool_autoscaler.controller.k8s_actuator import PodActuator, free_rollout_nodes, fully_free_rollout_nodes


class FakeGroup:
    def __init__(self, replicas=1):
        self._replicas = replicas
        self.patches = []
        self._pods = []

    def replicas(self):
        return self._replicas

    def set_replicas(self, n, workers_to_delete=None):
        self.patches.append((n, workers_to_delete))
        self._replicas = n

    def pods(self):
        return self._pods

    def pod_by_ip(self, ip):
        for p in self._pods:
            if p["status"]["podIP"] == ip:
                return p
        return None


def node(ip, total, avail, rollout=100):
    return {"ip": ip, "gpu_total": total, "gpu_available": avail, "rollout_node": rollout}


class NodeFilters(unittest.TestCase):
    def test_fallback_counts_plugin_replicas_when_ray_availability_unknown(self):
        st = {"nodes": [node("a", 1, None), node("b", 1, None)], "pool": {"active": [{"id": "standalone_0", "address": "a:35005"}], "parked": []}}
        self.assertEqual([n["ip"] for n in free_rollout_nodes(st)], ["b"])
        self.assertEqual([n["ip"] for n in fully_free_rollout_nodes(st)], ["b"])

    def test_free_and_fully_free(self):
        st = {"nodes": [node("a", 1, 0), node("b", 1, 1), {"ip": "head", "gpu_total": 0, "gpu_available": 0, "rollout_node": 0}]}
        self.assertEqual([n["ip"] for n in free_rollout_nodes(st)], ["b"])
        self.assertEqual([n["ip"] for n in fully_free_rollout_nodes(st)], ["b"])
        # unknown Ray availability + no plugin replicas on the node -> counted as free (fallback semantics)
        self.assertEqual([n["ip"] for n in free_rollout_nodes({"nodes": [node("a", 1, None)]})], ["a"])


class ScaleOutFlow(unittest.TestCase):
    def test_pod_added_then_ready_when_free_gpu_appears(self):
        g = FakeGroup(replicas=1)
        a = PodActuator(g, node_wait_timeout_s=100)
        rec = a.begin_scale_out(2, {"nodes": [node("a", 1, 0)]}, now=0.0)
        self.assertTrue(rec["pod_needed"])
        self.assertEqual(g.patches, [(2, None)])
        self.assertIsNone(a.poll_scale_out({"nodes": [node("a", 1, 0)]}, now=10.0))
        g._pods = [{"metadata": {"name": "p1"}, "status": {"phase": "Running", "podIP": "a"}}, {"metadata": {"name": "p2"}, "status": {"phase": "Running", "podIP": "b"}}]
        done = a.poll_scale_out({"nodes": [node("a", 1, 0), node("b", 1, 1)]}, now=40.0)
        self.assertTrue(done["ready"])
        self.assertEqual(done["node_wait_s"], 40.0 - rec["t_patch"])
        a.finish({"posted": {"ok": True}})
        self.assertIsNone(a.pending)
        self.assertEqual(a.history[-1]["posted"], {"ok": True})

    def test_no_pod_needed_when_spare_gpu(self):
        g = FakeGroup(replicas=2)
        a = PodActuator(g)
        rec = a.begin_scale_out(2, {"nodes": [node("b", 1, 1)]}, now=0.0)
        self.assertFalse(rec["pod_needed"])
        self.assertTrue(rec["ready"])
        self.assertEqual(g.patches, [])

    def test_timeout_records_error(self):
        g = FakeGroup(replicas=1)
        a = PodActuator(g, node_wait_timeout_s=5)
        a.begin_scale_out(2, {"nodes": [node("a", 1, 0)]}, now=0.0)
        done = a.poll_scale_out({"nodes": [node("a", 1, 0)]}, now=100.0)
        self.assertIn("error", done)
        self.assertIsNone(a.pending)


class ScaleInFlow(unittest.TestCase):
    def test_removes_the_pod_whose_gpu_became_free(self):
        g = FakeGroup(replicas=3)
        g._pods = [{"metadata": {"name": f"p{i}"}, "status": {"phase": "Running", "podIP": ip}} for i, ip in enumerate("abc")]
        a = PodActuator(g)
        a.begin_scale_in(2, now=0.0)
        # still 3 active: wait
        self.assertIsNone(a.poll_scale_in({"pool": {"active": [1, 2, 3]}, "nodes": [node("a", 1, 0), node("b", 1, 0), node("c", 1, 0)]}, now=5.0))
        # replica destroyed, node c fully free
        done = a.poll_scale_in({"pool": {"active": [1, 2]}, "nodes": [node("a", 1, 0), node("b", 1, 0), node("c", 1, 1)]}, now=20.0)
        self.assertEqual(done["pod_deleted"], "p2")
        self.assertEqual(g.patches, [(2, ["p2"])])
        self.assertIsNone(a.pending)


if __name__ == "__main__":
    unittest.main()
