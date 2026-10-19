import unittest

from rl_pool_autoscaler.metrics import MetricsTap


class MetricsTapTests(unittest.TestCase):
    def test_metrics_tap_on_stock_keys(self):
        class FakeTracking:
            def __init__(self):
                self.calls = []

            def log(self, data, step, backend=None):
                self.calls.append((data, step))

        inner = FakeTracking()
        tap = MetricsTap(inner)
        # what _fit_update_weights logs (rollouter.reset_staleness timing)
        tap.log(data={"fully_async/rollouter/idle_ratio": 0.3, "fully_async/rollouter/version_time": 120.0}, step=4)
        # what _fit_log_aggregated_training_metrics logs (MetricsAggregator output)
        tap.log(
            data={
                "fully_async/total_wait_time": 300.0,
                "timing_s/step": 480.0,
                "timing_s/timing_s/param_sync": 6.5,
                "dynamic_resource/mq_size": 12,
                "fully_async/count/current_param_version": 4,
                "training/global_step": 16,
            },
            step=4,
        )
        snap = tap.control_snapshot()
        self.assertEqual(len(inner.calls), 2)  # forwarded unchanged
        self.assertAlmostEqual(snap["trainer_idle_ratio"], 300.0 / 480.0)
        self.assertEqual(snap["last_sync_wall_s"], 6.5)
        self.assertEqual(snap["mq_size"], 12)
        self.assertEqual(snap["rollouter_idle_ratio"], 0.3)
        self.assertEqual(snap["log_step"], 4)


    def test_non_numeric_and_bad_inner_do_not_break(self):
        class Inner:
            def log(self, data, step, backend=None):
                return "forwarded"

            def finish(self, exit_code=0):
                return "finished"

        tap = MetricsTap(Inner())
        self.assertEqual(tap.log(data={"text": "x", "n": 1}, step=1), "forwarded")
        self.assertEqual(tap.latest, {"n": 1.0})
        self.assertEqual(tap.finish(), "finished")  # attribute passthrough


if __name__ == "__main__":
    unittest.main()


class TrainLogStep(unittest.TestCase):
    def test_only_training_logs_advance_train_log_step(self):
        from rl_pool_autoscaler.metrics import MetricsTap

        class Inner:
            def log(self, data, step, backend=None):
                return None

        tap = MetricsTap(Inner())
        tap.log({"fully_async/rollouter/idle_ratio": 0.0, "dynamic_resource/rollout_resource_utilization": 0.6}, step=9)
        self.assertEqual(tap.control_snapshot()["log_step"], 9)
        self.assertIsNone(tap.control_snapshot()["train_log_step"])
        tap.log({"timing_s/step": 600.0, "fully_async/total_wait_time": 550.0}, step=9)
        self.assertEqual(tap.control_snapshot()["train_log_step"], 9)
