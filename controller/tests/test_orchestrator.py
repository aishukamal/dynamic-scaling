import unittest

from fakes import FakeBackend, prestarted

from rl_pool_autoscaler.orchestrator import PoolLimits, ScaleError, ScaleOrchestrator


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make(backend, min_=1, max_=4, **kw):
    clock = FakeClock()
    kw.setdefault("log", None)
    return ScaleOrchestrator(backend, PoolLimits(min_, max_), clock=clock, sleep=clock.sleep, **kw), clock


class ScaleOutOrdering(unittest.TestCase):
    def test_scale_out_order_from_parked(self):
        b = prestarted(4, 1)
        o, _ = make(b)
        o.request(2)
        self.assertEqual(o.converge(), ["scale_out"])
        rid = "standalone_1"
        self.assertEqual(b.ops_for(rid), ["wake", "attach", "sync_joiners", "resume_generation", "lb_add"])
        self.assertEqual(b.active, ["standalone_0", "standalone_1"])
        self.assertEqual(b.parked, ["standalone_2", "standalone_3"])
        self.assertNotIn(("create_standalone", ["standalone_100"]), b.calls)

    def test_scale_out_creates_when_nothing_parked(self):
        b = prestarted(1, 1)
        o, _ = make(b, 1, 3)
        o.request(2)
        o.converge()
        self.assertEqual([op for op, _ in b.calls], ["create_standalone", "wake", "attach", "sync_joiners", "resume_generation", "lb_add"])
        self.assertEqual(b.active, ["standalone_0", "standalone_100"])

    def test_lb_add_never_precedes_sync(self):
        b = prestarted(4, 1)
        o, _ = make(b)
        o.request(4)
        o.converge()
        for rid in ("standalone_1", "standalone_2", "standalone_3"):
            ops = b.ops_for(rid)
            self.assertLess(ops.index("attach"), ops.index("sync_joiners"))
            self.assertLess(ops.index("sync_joiners"), ops.index("lb_add"))

    def test_deferred_joiner_sync_waits_for_regular_sync(self):
        b = prestarted(2, 1)
        b.defer_joiner_sync = True  # trainer weights dirty: joiner must ride the next regular sync
        o, clock = make(b)
        o.request(2)
        # emulate the regular sync landing after 3 polls
        polls = {"n": 0}
        real_sleep = clock.sleep

        def sleep(s):
            real_sleep(s)
            polls["n"] += 1
            if polls["n"] == 3:
                b.regular_sync()

        o.sleep = sleep
        self.assertEqual(o.converge(), ["scale_out"])
        self.assertEqual(b.active, ["standalone_0", "standalone_1"])
        self.assertEqual(b.versions["standalone_1"], b.last_synced)
        self.assertGreaterEqual(b.ops_for("standalone_1").count("sync_joiners"), 2)
        self.assertEqual(b.ops_for("standalone_1")[-1], "lb_add")

    def test_gate_timeout_rolls_back(self):
        b = prestarted(2, 1)
        b.defer_joiner_sync = True
        o, _ = make(b, join_timeout_s=20, join_poll_s=5)
        o.request(2)
        with self.assertRaises(ScaleError):
            o.step()
        ops = b.ops_for("standalone_1")
        self.assertNotIn("lb_add", ops)
        self.assertEqual(ops[-2:], ["detach", "sleep"])
        self.assertEqual(b.active, ["standalone_0"])
        self.assertEqual(b.attached, ["standalone_0"])
        self.assertEqual(o.ops_total["rollback"], 1)
        self.assertIn("version gate", o.last_error)

    def test_stale_joiner_not_registered_when_pool_moves_ahead(self):
        # joiner synced at v0, then a regular sync advances the pool to v1 BEFORE lb_add is
        # evaluated: gate must re-sync rather than register a stale replica.
        b = prestarted(2, 1)
        o, _ = make(b)
        orig = b.sync_joiners
        state = {"n": 0}

        def sync_then_advance(rids):
            res = orig(rids)
            state["n"] += 1
            if state["n"] == 1:
                b.trainer_version += 1  # trainer bumps version
                b.last_synced = b.trainer_version
                for r in b.attached:
                    if r not in rids:
                        b.versions[r] = b.trainer_version
                # joiner stays at old version -> gate must fail once
            return res

        b.sync_joiners = sync_then_advance
        o.request(2)
        o.converge()
        self.assertEqual(b.versions["standalone_1"], b.last_synced)
        self.assertEqual(b.ops_for("standalone_1").count("sync_joiners"), 2)


class ScaleInOrdering(unittest.TestCase):
    def test_scale_in_order_sleep_tier(self):
        b = prestarted(4, 3)
        o, _ = make(b)
        o.request(2)
        self.assertEqual(o.converge(), ["scale_in"])
        self.assertEqual(b.ops_for("standalone_2"), ["lb_remove", "abort", "detach", "sleep"])
        self.assertEqual(b.active, ["standalone_0", "standalone_1"])
        self.assertIn("standalone_2", b.parked)

    def test_scale_in_release_destroys(self):
        b = prestarted(3, 3)
        o, _ = make(b, release_on_scale_in=True)
        o.request(2)
        o.converge()
        self.assertEqual(b.ops_for("standalone_2"), ["lb_remove", "abort", "detach", "destroy"])
        self.assertEqual(b.destroyed, ["standalone_2"])
        self.assertNotIn("standalone_2", b.parked)

    def test_release_flag_via_request(self):
        b = prestarted(3, 3)
        o, _ = make(b)
        o.request(2, release=True)
        o.converge()
        self.assertEqual(b.destroyed, ["standalone_2"])

    def test_never_below_min(self):
        b = prestarted(4, 2)
        o, _ = make(b, 2, 4)
        res = o.request(0)
        self.assertEqual(res["desired"], 2)
        self.assertTrue(res["clamped"])
        self.assertEqual(o.converge(), [])
        self.assertEqual(b.active, ["standalone_0", "standalone_1"])
        # direct call is also refused
        with self.assertRaises(ScaleError):
            o.scale_in_one()

    def test_never_above_max(self):
        b = prestarted(4, 1)
        o, _ = make(b, 1, 3)
        res = o.request(10)
        self.assertEqual(res["desired"], 3)
        o.converge()
        self.assertEqual(len(b.active), 3)
        with self.assertRaises(ScaleError):
            o.scale_out_one()


class Idempotency(unittest.TestCase):
    def test_same_request_twice_is_noop(self):
        b = prestarted(4, 1)
        o, _ = make(b)
        o.request(3)
        o.converge()
        n_calls = len(b.calls)
        r = o.request(3)
        self.assertTrue(r["noop"])
        self.assertEqual(o.converge(), [])
        self.assertEqual(len(b.calls), n_calls)
        self.assertEqual(o.ops_total["noop"], 1)

    def test_request_equal_to_active_is_noop(self):
        b = prestarted(4, 2)
        o, _ = make(b)
        r = o.request(2)
        self.assertTrue(r["noop"])
        self.assertEqual(o.converge(), [])

    def test_one_move_per_step_and_retarget_midway(self):
        b = prestarted(4, 1)
        o, _ = make(b)
        o.request(4)
        self.assertEqual(o.step(), "scale_out")
        self.assertEqual(len(b.active), 2)
        o.request(2)  # changed its mind
        self.assertIsNone(o.step())
        self.assertEqual(len(b.active), 2)

    def test_round_trip_returns_to_warm_pool(self):
        b = prestarted(3, 1)
        o, _ = make(b)
        o.request(3)
        o.converge()
        o.request(1)
        o.converge()
        self.assertEqual(b.active, ["standalone_0"])
        self.assertEqual(sorted(b.parked), ["standalone_1", "standalone_2"])
        self.assertEqual(b.attached, ["standalone_0"])
        self.assertEqual(o.ops_total, {"scale_out": 2, "scale_in": 2, "rollback": 0, "noop": 0, "evict": 0})

    def test_failure_is_recorded_and_in_progress_cleared(self):
        b = prestarted(2, 1)
        b.fail_on = "wake"
        o, _ = make(b)
        o.request(2)
        with self.assertRaises(RuntimeError):
            o.step()
        self.assertFalse(o.in_progress)
        self.assertIn("injected failure", o.last_error)
        self.assertEqual(o.status()["recent_events"][-1]["action"], "error")


class Limits(unittest.TestCase):
    def test_min_must_be_at_least_one(self):
        with self.assertRaises(ValueError):
            PoolLimits(0, 4)
        with self.assertRaises(ValueError):
            PoolLimits(3, 2)


if __name__ == "__main__":
    unittest.main()


class EvictDead(unittest.TestCase):
    """Failure path: a dead ACTIVE replica is cut from LB + membership + destroyed, then replaced by step()."""

    def test_evict_then_replace_from_parked(self):
        b = FakeBackend(active=["standalone_0", "standalone_1"], parked=["standalone_2"], attached=["standalone_0", "standalone_1"],
                        versions={"standalone_0": 3, "standalone_1": 3, "standalone_2": 3}, last_synced=3, trainer_version=3)
        o = ScaleOrchestrator(b, PoolLimits(1, 4), log=None)
        o.request(2)
        self.assertEqual(o.evict_dead("standalone_1", {"detected_at": 1.0}), "evict")
        ops = [c[0] for c in b.calls]
        self.assertEqual(ops[-3:], ["evict", "detach", "destroy"])
        self.assertEqual(b.active, ["standalone_0"])
        self.assertIn("standalone_1", b.destroyed)
        self.assertEqual(o.ops_total["evict"], 1)
        self.assertEqual(o.step(), "scale_out")  # desired still 2 -> parked standalone_2 joins
        self.assertEqual(b.active, ["standalone_0", "standalone_2"])
        self.assertEqual(o.events[-1].action, "scale_out")
        self.assertEqual(o.events[-1].detail["source"], "wake")

    def test_evict_survives_backend_errors(self):
        b = FakeBackend(active=["standalone_0", "standalone_1"], attached=["standalone_0", "standalone_1"], versions={"standalone_0": 1, "standalone_1": 1}, last_synced=1, trainer_version=1, fail_on="destroy")
        o = ScaleOrchestrator(b, PoolLimits(1, 4), log=None)
        o.evict_dead("standalone_1")
        self.assertEqual(b.active, ["standalone_0"])
        self.assertNotIn("standalone_1", b.attached)
        self.assertIn("destroy", o.events[-1].detail["errors"])
