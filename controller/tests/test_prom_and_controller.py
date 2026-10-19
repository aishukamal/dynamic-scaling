import json
import os
import tempfile
import unittest

from rl_pool_autoscaler import prom
from rl_pool_autoscaler._compat import RLPoolConfig, rl_pool_config, unwrap_actor_class
from rl_pool_autoscaler.controller.main import DecisionLog, observation_from_samples, run_once
from rl_pool_autoscaler.controller.policy import BandPolicy, PolicyConfig


class PromRoundTrip(unittest.TestCase):
    def test_render_parse(self):
        text = prom.render(
            [
                ("a_gauge", 1.5, None, "help a", "gauge"),
                ("a_gauge", 2, {"replica": "standalone_1", "address": "10.0.0.1:8000"}, "help a", "gauge"),
                ("b_bool", True, None, None, "gauge"),
                ("c_none", None, None, None, "gauge"),
            ]
        )
        self.assertIn("# HELP a_gauge help a", text)
        self.assertEqual(text.count("# TYPE a_gauge"), 1)
        s = prom.parse(text)
        self.assertEqual(prom.scalar(s, "a_gauge"), 1.5)
        self.assertEqual(prom.by_label(s, "a_gauge", "replica"), {"standalone_1": 2.0})
        self.assertEqual(prom.scalar(s, "b_bool"), 1.0)
        self.assertNotEqual(prom.scalar(s, "c_none"), prom.scalar(s, "c_none"))  # NaN

    def test_observation_from_samples(self):
        text = "\n".join(
            [
                "rl_pool_current_param_version 12",
                "rl_pool_trainer_idle_ratio 0.4",
                "rl_pool_mq_size 30",
                "rl_pool_max_queue_size 160",
                "rl_pool_active_standalone_replicas 2",
                "rl_pool_desired_standalone_replicas 2",
                "rl_pool_scaling_in_progress 0",
                'rl_pool_replica_inflight_requests{replica="standalone_0",address="x"} 5',
            ]
        )
        o = observation_from_samples(prom.parse(text), now=1.0)
        self.assertEqual((o.param_version, o.active_standalone, o.desired_standalone), (12, 2, 2))
        self.assertEqual(o.extra["inflight"], {"standalone_0": 5.0})
        self.assertIsNone(observation_from_samples(prom.parse("rl_pool_mq_size 1"), now=1.0))


class FakeClient:
    def __init__(self, texts):
        self.texts = list(texts)
        self.posted = []
        self.nodes = []

    def metrics(self):
        return prom.parse(self.texts.pop(0))

    def scale(self, standalone, release=None):
        self.posted.append(standalone)
        out = {"desired": standalone, "noop": False}
        if release is not None:
            out["release"] = release
        return out

    def status(self):
        return {"pool": {"active": [{"id": "standalone_0"}]}, "nodes": self.nodes}


def metrics_text(version, idle, mq=10, active=1):
    return "\n".join(
        [
            f"rl_pool_current_param_version {version}",
            f"rl_pool_trainer_idle_ratio {idle}",
            f"rl_pool_mq_size {mq}",
            "rl_pool_max_queue_size 160",
            f"rl_pool_active_standalone_replicas {active}",
            f"rl_pool_desired_standalone_replicas {active}",
            "rl_pool_scaling_in_progress 0",
        ]
    )


class NoSampleBeforeFirstTrainingLog(unittest.TestCase):
    def test_version_without_step_metrics_is_not_a_sample(self):
        from rl_pool_autoscaler.controller.main import observation_from_samples

        txt = "rl_pool_current_param_version 1\nrl_pool_trainer_train_log_step NaN\nrl_pool_trainer_step_s NaN\nrl_pool_active_standalone_replicas 1\nrl_pool_rollout_resource_utilization 0.0\n"
        self.assertIsNone(observation_from_samples(prom.parse(txt), now=1.0))
        txt2 = txt.replace("rl_pool_trainer_train_log_step NaN", "rl_pool_trainer_train_log_step 1").replace("rl_pool_trainer_step_s NaN", "rl_pool_trainer_step_s 500")
        self.assertEqual(observation_from_samples(prom.parse(txt2), now=1.0).param_version, 1)


class ControllerLoop(unittest.TestCase):
    def test_run_once_logs_jsonl_and_posts(self):
        client = FakeClient([metrics_text(1, 0.8), metrics_text(2, 0.8), metrics_text(2, 0.8)])
        policy = BandPolicy(PolicyConfig(min_standalone=1, max_standalone=4, dwell_steps=2, min_action_interval_s=0))
        base = os.path.expanduser("~/workspaces")
        os.makedirs(base, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as d:
            path = os.path.join(d, "decisions.jsonl")
            log = DecisionLog(path)
            run_once(policy, client, log, dry_run=False, now=100.0)
            run_once(policy, client, log, dry_run=False, now=400.0)
            run_once(policy, client, log, dry_run=False, now=410.0)
            log.close()
            with open(path) as fh:
                recs = [json.loads(line) for line in fh]
        self.assertEqual(len(recs), 3)
        self.assertEqual(recs[0]["decision"]["action"], "hold")
        self.assertEqual(recs[1]["decision"]["action"], "scale_out")
        self.assertEqual(recs[1]["posted"], {"desired": 2, "noop": False})
        self.assertEqual(recs[2]["decision"]["reason"], "no new param version")
        self.assertEqual(client.posted, [2])
        self.assertIn("trainer_idle_ratio", recs[1]["decision"]["inputs"])

    def test_dry_run_never_posts(self):
        client = FakeClient([metrics_text(1, 0.8), metrics_text(2, 0.8)])
        policy = BandPolicy(PolicyConfig(min_standalone=1, max_standalone=4, dwell_steps=2, min_action_interval_s=0))
        log = DecisionLog("")
        run_once(policy, client, log, dry_run=True, now=1.0)
        rec = run_once(policy, client, log, dry_run=True, now=2.0)
        self.assertEqual(rec["posted"], {"dry_run": True, "standalone": 2})
        self.assertEqual(client.posted, [])


class CompatHelpers(unittest.TestCase):
    def test_unwrap_actor_class(self):
        class Orig:
            pass

        class Modified(Orig):
            __ray_actor_class__ = Orig

        class Meta:
            modified_class = Modified

        class ActorClassLike:
            __ray_metadata__ = Meta()

        self.assertIs(unwrap_actor_class(ActorClassLike), Orig)
        self.assertIs(unwrap_actor_class(Orig), Orig)

    def test_rl_pool_config(self):
        self.assertEqual(rl_pool_config({}), RLPoolConfig())
        c = rl_pool_config({"rl_pool": {"min_standalone": 2, "max_standalone": 6, "prestart": False}})
        self.assertEqual((c.min_standalone, c.max_standalone, c.prestart), (2, 6, False))
        with self.assertRaises(ValueError):
            rl_pool_config({"rl_pool": {"min_standalone": 0}})
        with self.assertRaises(ValueError):
            rl_pool_config({"rl_pool": {"bogus": 1}})
        with self.assertRaises(ValueError):
            rl_pool_config({"rl_pool": {"joiner_sync_mode": "whenever"}})


if __name__ == "__main__":
    unittest.main()
