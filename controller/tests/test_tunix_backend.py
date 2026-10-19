"""Tunix backend tests: JobSet actuator with a fake kube client and fake /status, gain guard, controller loop."""

import json
import os
import tempfile
import unittest

from rl_pool_autoscaler.controller.policy import BandPolicy, Observation, PolicyConfig
from rl_pool_autoscaler.controller.tunix_backend import GainGuard, JobSetActuator, SlotUtilTracker, Template, TunixStatus
from rl_pool_autoscaler.controller.tunix_main import DecisionLog, TunixController


def status(step, wait, wall=50.0, workers=None, retries=0, evictions=0, history=None, sip=None, cap=128, pending=0, t=None):
    """Build a /status JSON like tunix_exporter.Exporter.snapshot() and parse it."""
    workers = workers or {"tc-roll-0": ("active", 1, 100)}
    d = {
        "t": (1000.0 + step * 50) if t is None else t,
        "last_completed_step": step,
        "step_in_progress": sip if sip is not None else step + 1,
        "step": {"step": step, "step_time_sec": wall, "exposed_generation_time": wait, "trainer_wait_ratio": wait / wall},
        "workers": {w: {"state": s, "incarnation": inc, "inflight": infl, "max_concurrency": cap, "state_times": {}}
                    for w, (s, inc, infl) in workers.items()},
        "counts": {},
        "rollout_queue": {"pending": pending, "inflight_total": sum(x[2] for x in workers.values())},
        "fault_tolerance": {"rollout_retries_total": retries, "rollout_worker_evictions_total": evictions},
        "history": history if history is not None else [{"step": s, "exposed_generation_time": wait} for s in range(max(0, step - 5), step + 1)],
    }
    return TunixStatus.from_json(d)


class FakeKube:
    def __init__(self):
        self.applied = []
        self.deleted = []

    def apply(self, manifest):
        self.applied.append(manifest)
        return "applied"

    def delete_jobset(self, name):
        self.deleted.append(name)
        return "deleted"

    def jobset_exists(self, name):
        return False


def templates():
    return [Template(0, "tc-roll-1", "tc-roll-1", "kind: JobSet\nname: tc-roll-1\n"),
            Template(1, "tc-roll-2", "tc-roll-2", "kind: JobSet\nname: tc-roll-2\n")]


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class ActuatorScaleOut(unittest.TestCase):
    def test_apply_then_active_completes_with_join_latency(self):
        kube, clk = FakeKube(), Clock(100.0)
        a = JobSetActuator(kube, templates(), {"tc-roll-0"}, join_timeout_s=300, clock=clk)
        st = status(2, 25.0)
        self.assertTrue(a.can_scale_out(st))
        op = a.scale_out(st)
        self.assertEqual(op["worker_id"], "tc-roll-1")
        self.assertEqual(kube.applied, ["kind: JobSet\nname: tc-roll-1\n"])
        self.assertTrue(a.in_progress)
        self.assertFalse(a.can_scale_out(st))
        clk.t = 140.0
        self.assertIsNone(a.tick(status(2, 25.0, workers={"tc-roll-0": ("active", 1, 100), "tc-roll-1": ("initializing", 1, 0)})))
        clk.t = 190.0
        self.assertIsNone(a.tick(status(3, 25.0, workers={"tc-roll-0": ("active", 1, 100), "tc-roll-1": ("pending_weight_sync", 1, 0)})))
        clk.t = 198.0
        done = a.tick(status(3, 25.0, workers={"tc-roll-0": ("active", 1, 100), "tc-roll-1": ("active", 1, 0)}))
        self.assertEqual(done["result"], "ok")
        self.assertAlmostEqual(done["join_latency_s"], 98.0)
        self.assertFalse(a.in_progress)
        # next scale-out goes to the next template
        st2 = status(5, 11.0, workers={"tc-roll-0": ("active", 1, 60), "tc-roll-1": ("active", 1, 60)})
        self.assertEqual(a.next_template(st2).worker_id, "tc-roll-2")

    def test_rejoin_requires_new_incarnation(self):
        kube, clk = FakeKube(), Clock(0.0)
        a = JobSetActuator(kube, templates(), {"tc-roll-0"}, clock=clk)
        # roll-1 exists as EVICTED incarnation 1 (removed earlier); scale-out must wait for incarnation 2 ACTIVE
        st = status(10, 25.0, workers={"tc-roll-0": ("active", 1, 100), "tc-roll-1": ("evicted", 1, 0)})
        self.assertEqual(a.next_template(st).worker_id, "tc-roll-1")
        a.scale_out(st)
        clk.t = 50
        self.assertIsNone(a.tick(status(10, 25.0, workers={"tc-roll-0": ("active", 1, 100), "tc-roll-1": ("evicted", 1, 0)})))
        done = a.tick(status(11, 25.0, workers={"tc-roll-0": ("active", 1, 100), "tc-roll-1": ("active", 2, 0)}))
        self.assertEqual(done["result"], "ok")
        self.assertEqual(done["incarnation"], 2)

    def test_timeout_deletes_jobset_and_marks_template_failed(self):
        kube, clk = FakeKube(), Clock(0.0)
        a = JobSetActuator(kube, templates(), {"tc-roll-0"}, join_timeout_s=300, clock=clk)
        a.scale_out(status(2, 25.0))
        clk.t = 299
        self.assertIsNone(a.tick(status(5, 25.0, workers={"tc-roll-0": ("active", 1, 100), "tc-roll-1": ("initializing", 1, 0)})))
        clk.t = 301
        done = a.tick(status(5, 25.0, workers={"tc-roll-0": ("active", 1, 100), "tc-roll-1": ("initializing", 1, 0)}))
        self.assertEqual(done["result"], "failed")
        self.assertEqual(kube.deleted, ["tc-roll-1"])
        self.assertIn("tc-roll-1", a.failed_templates)
        self.assertEqual(a.next_template(status(6, 25.0)).worker_id, "tc-roll-2")


class ActuatorScaleIn(unittest.TestCase):
    def test_picks_fewest_inflight_never_base_and_counts_requeues(self):
        kube, clk = FakeKube(), Clock(0.0)
        a = JobSetActuator(kube, templates(), {"tc-roll-0"}, evict_timeout_s=100, clock=clk)
        st = status(20, 5.0, workers={"tc-roll-0": ("active", 1, 2), "tc-roll-1": ("active", 1, 40), "tc-roll-2": ("active", 1, 7)}, retries=100)
        self.assertTrue(a.can_scale_in(st))
        op = a.scale_in(st)
        self.assertEqual(op["worker_id"], "tc-roll-2")  # base roll-0 has fewer in flight but is never a candidate
        self.assertEqual(kube.deleted, ["tc-roll-2"])
        clk.t = 3
        self.assertIsNone(a.tick(status(20, 5.0, workers={"tc-roll-0": ("active", 1, 2), "tc-roll-1": ("active", 1, 40), "tc-roll-2": ("active", 1, 7)}, retries=100)))
        clk.t = 6
        done = a.tick(status(20, 5.0, workers={"tc-roll-0": ("active", 1, 2), "tc-roll-1": ("active", 1, 40), "tc-roll-2": ("evicted", 1, 0)}, retries=107, evictions=1))
        self.assertEqual(done["result"], "ok")
        self.assertEqual(done["requeued"], 7)
        self.assertEqual(done["evictions_delta"], 1)
        self.assertAlmostEqual(done["evict_latency_s"], 6.0)

    def test_tie_prefers_newest(self):
        a = JobSetActuator(FakeKube(), templates(), {"tc-roll-0"})
        st = status(20, 5.0, workers={"tc-roll-0": ("active", 1, 9), "tc-roll-1": ("active", 1, 9), "tc-roll-2": ("active", 1, 9)})
        self.assertEqual(a.pick_scale_in(st).worker_id, "tc-roll-2")

    def test_only_base_left_cannot_scale_in(self):
        a = JobSetActuator(FakeKube(), templates(), {"tc-roll-0"})
        self.assertFalse(a.can_scale_in(status(3, 5.0)))


class Guard(unittest.TestCase):
    def test_blocks_when_wait_did_not_drop_and_unblocks_on_scale_in(self):
        g = GainGuard(window=2, min_gain=0.10)
        hist = [{"step": s, "exposed_generation_time": 11.0} for s in range(0, 10)]
        st = status(9, 11.0, history=hist, workers={"tc-roll-0": ("active", 1, 1), "tc-roll-1": ("active", 1, 1)})
        g.on_scale_out_issued(st, "tc-roll-2")
        self.assertEqual(g.pending["before_steps"], [8, 9])
        st_join = status(10, 11.0, history=hist + [{"step": 10, "exposed_generation_time": 11.0}], sip=11,
                         workers={"tc-roll-0": ("active", 1, 1), "tc-roll-1": ("active", 1, 1), "tc-roll-2": ("active", 1, 0)})
        g.on_join_done(st_join, {"worker_id": "tc-roll-2", "result": "ok", "kind": "scale_out"})
        self.assertEqual(g.pending["first_clean_step"], 12)
        self.assertIsNone(g.update(status(12, 11.0, history=hist + [{"step": s, "exposed_generation_time": 11.0} for s in (10, 11, 12)])))
        ev = g.update(status(13, 11.0, history=hist + [{"step": s, "exposed_generation_time": 11.0} for s in (10, 11, 12, 13)]))
        self.assertTrue(ev["blocked"])
        self.assertAlmostEqual(ev["gain"], 0.0)
        self.assertEqual(ev["after_steps"], [12, 13])
        self.assertTrue(g.blocked)
        g.on_scale_in()
        self.assertFalse(g.blocked)

    def test_passes_when_wait_drops(self):
        g = GainGuard(window=2, min_gain=0.10)
        hist = [{"step": s, "exposed_generation_time": 26.0} for s in range(0, 4)]
        g.on_scale_out_issued(status(3, 26.0, history=hist), "tc-roll-1")
        g.on_join_done(status(4, 26.0, sip=5), {"worker_id": "tc-roll-1", "result": "ok", "kind": "scale_out"})
        h2 = hist + [{"step": 4, "exposed_generation_time": 26.0}, {"step": 5, "exposed_generation_time": 18.0},
                     {"step": 6, "exposed_generation_time": 11.0}, {"step": 7, "exposed_generation_time": 11.0}]
        ev = g.update(status(7, 11.0, history=h2))
        self.assertFalse(ev["blocked"])
        self.assertGreater(ev["gain"], 0.5)


class ScriptedClient:
    def __init__(self, seq):
        self.seq = list(seq)
        self.i = 0

    def get(self):
        st = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return st


class ControllerLoop(unittest.TestCase):
    def _cfg(self):
        return PolicyConfig(min_standalone=1, max_standalone=3, idle_hi=0.40, idle_lo=0.15, dwell_steps=2,
                            min_action_interval_s=120, mq_lo_frac=0.8, mq_hi_frac=0.0)

    def test_one_to_two_then_hold(self):
        W1 = {"tc-roll-0": ("active", 1, 128)}
        W1j = {"tc-roll-0": ("active", 1, 128), "tc-roll-1": ("initializing", 1, 0)}
        W2 = {"tc-roll-0": ("active", 1, 64), "tc-roll-1": ("active", 1, 64)}
        seq = [status(0, 14.8, wall=109.0, workers=W1),      # compile step, ratio 0.14 -> hold
               status(1, 25.5, wall=53.0, workers=W1),       # 0.48 hi_count 1
               status(2, 25.6, wall=53.0, workers=W1),       # 0.48 hi_count 2 -> scale_out
               status(2, 25.6, wall=53.0, workers=W1j),      # in progress
               status(3, 25.4, wall=53.0, workers=W1j),      # in progress (new step, hold)
               status(4, 25.4, wall=53.0, workers=W2, sip=5),  # ACTIVE -> join done
               status(5, 20.0, wall=45.0, workers=W2),       # contaminated step
               status(6, 11.0, wall=38.5, workers=W2),       # 0.29 in band
               status(7, 11.0, wall=38.5, workers=W2),
               status(8, 11.0, wall=38.5, workers=W2)]
        kube = FakeKube()
        clk = Clock(0.0)
        act = JobSetActuator(kube, templates(), {"tc-roll-0"}, clock=clk)
        guard = GainGuard(2, 0.10)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dec.jsonl")
            ctl = TunixController(BandPolicy(self._cfg()), ScriptedClient(seq), act, guard, DecisionLog(path))
            recs = []
            for k in range(len(seq)):
                clk.t = k * 50.0
                recs.append(ctl.run_once(now=clk.t))
            lines = [json.loads(l) for l in open(path)]
        self.assertEqual(len(lines), len(seq))
        actions = [(r["decision"]["action"], r["decision"]["inputs"]["param_version"]) for r in recs if "decision" in r]
        self.assertIn(("scale_out", 2), actions)
        self.assertEqual(sum(1 for a, _ in actions if a == "scale_out"), 1)
        self.assertEqual(kube.applied[0], "kind: JobSet\nname: tc-roll-1\n")
        self.assertEqual(kube.deleted, [])
        done = [r["actuation_done"] for r in recs if "actuation_done" in r]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["result"], "ok")
        self.assertEqual(done[0]["active_after"], 2)
        # guard evaluated on the first two clean steps after the join (6, 7): wait 25.5 -> 11.0, not blocked
        ev = [r["guard_eval"] for r in recs if "guard_eval" in r]
        self.assertEqual(len(ev), 1)
        self.assertFalse(ev[0]["blocked"])
        self.assertEqual(ev[0]["after_steps"], [6, 7])
        # trajectory records each new step with the active count
        self.assertEqual([(t["step"], t["active"]) for t in ctl.trajectory][-3:], [(6, 2), (7, 2), (8, 2)])
        # hold reasons while in progress
        reasons = [r["decision"]["reason"] for r in recs if "decision" in r]
        self.assertTrue(any("scaling in progress" in x for x in reasons))

    def test_guard_blocks_third_worker_when_no_gain(self):
        W2 = {"tc-roll-0": ("active", 1, 64), "tc-roll-1": ("active", 1, 64)}
        W2j = dict(W2, **{"tc-roll-2": ("initializing", 1, 0)})
        W3 = {"tc-roll-0": ("active", 1, 43), "tc-roll-1": ("active", 1, 43), "tc-roll-2": ("active", 1, 42)}
        # band deliberately set so that 0.29 counts as "high": verifies the guard, not the band
        cfg = PolicyConfig(min_standalone=1, max_standalone=4, idle_hi=0.25, idle_lo=0.05, dwell_steps=2,
                           min_action_interval_s=0, mq_hi_frac=0.0)
        seq = [status(10, 11.0, wall=38.5, workers=W2), status(11, 11.0, wall=38.5, workers=W2),  # -> scale_out roll-2
               status(11, 11.0, wall=38.5, workers=W2j), status(12, 11.0, wall=38.5, workers=W3, sip=13),  # join done
               status(13, 11.0, wall=38.5, workers=W3), status(14, 11.0, wall=38.5, workers=W3),
               status(15, 11.0, wall=38.5, workers=W3),   # guard evaluates on 14,15 -> no gain -> blocked
               status(16, 11.0, wall=38.5, workers=W3), status(17, 11.0, wall=38.5, workers=W3)]
        t3 = templates() + [Template(2, "tc-roll-3", "tc-roll-3", "kind: JobSet\nname: tc-roll-3\n")]
        kube, clk = FakeKube(), Clock(0.0)
        act = JobSetActuator(kube, t3, {"tc-roll-0"}, clock=clk)
        ctl = TunixController(BandPolicy(cfg), ScriptedClient(seq), act, GainGuard(2, 0.10), DecisionLog(""))
        recs = []
        for k in range(len(seq)):
            clk.t = k * 50.0
            recs.append(ctl.run_once(now=clk.t))
        self.assertEqual(len(kube.applied), 1)  # roll-2 only; roll-3 blocked
        blocked = [r for r in recs if r.get("actuation", {}).get("result") == "blocked_by_guard"]
        self.assertGreaterEqual(len(blocked), 1)
        ev = [r["guard_eval"] for r in recs if "guard_eval" in r][0]
        self.assertTrue(ev["blocked"])
        self.assertEqual(ev["before_wait_s"], 11.0)
        self.assertEqual(ev["after_wait_s"], 11.0)

    def test_dry_run_client_error_is_logged(self):
        class Boom:
            def get(self):
                raise OSError("connection refused")
        ctl = TunixController(BandPolicy(self._cfg()), Boom(), JobSetActuator(FakeKube(), templates()), None, DecisionLog(""))
        rec = ctl.run_once(now=1.0)
        self.assertIn("error", rec)


class PolicyQueueNeutralized(unittest.TestCase):
    def test_mq_hi_zero_lets_band_scale_in_without_queue_gauge(self):
        p = BandPolicy(PolicyConfig(min_standalone=1, max_standalone=3, idle_hi=0.40, idle_lo=0.15, dwell_steps=2,
                                    min_action_interval_s=0, mq_hi_frac=0.0))
        d1 = p.observe(Observation(t=0, param_version=1, trainer_idle_ratio=0.05, mq_size=0, max_queue_size=0, active_standalone=2))
        d2 = p.observe(Observation(t=1, param_version=2, trainer_idle_ratio=0.05, mq_size=0, max_queue_size=0, active_standalone=2))
        self.assertEqual((d1.action, d2.action), ("hold", "scale_in"))
        d3 = p.observe(Observation(t=2, param_version=3, trainer_idle_ratio=0.48, mq_size=0, max_queue_size=0, active_standalone=1))
        d4 = p.observe(Observation(t=3, param_version=4, trainer_idle_ratio=0.48, mq_size=0, max_queue_size=0, active_standalone=1))
        self.assertEqual((d3.action, d4.action), ("hold", "scale_out"))


if __name__ == "__main__":
    unittest.main()


class FailureActBehaviour(unittest.TestCase):
    """A pod of an ACTIVE worker dies; the JobSet recreates it; the worker is EVICTED then rejoins (incarnation 2).
    The controller must not add a third worker while the rejoin is in flight, and must restart dwell when the
    pool changes under it."""

    def test_holds_through_evict_and_rejoin(self):
        cfg = PolicyConfig(min_standalone=1, max_standalone=3, idle_hi=0.40, idle_lo=0.15, dwell_steps=2,
                           min_action_interval_s=0, mq_hi_frac=0.0)
        W2 = {"tc-roll-0": ("active", 1, 64), "tc-roll-1": ("active", 1, 64)}
        W_ev = {"tc-roll-0": ("active", 1, 128), "tc-roll-1": ("evicted", 1, 0)}
        W_init = {"tc-roll-0": ("active", 1, 128), "tc-roll-1": ("initializing", 2, 0)}
        W_pend = {"tc-roll-0": ("active", 1, 128), "tc-roll-1": ("pending_weight_sync", 2, 0)}
        W2b = {"tc-roll-0": ("active", 1, 64), "tc-roll-1": ("active", 2, 64)}
        seq = [status(10, 11.0, wall=38.5, workers=W2), status(11, 11.0, wall=38.5, workers=W2),
               status(11, 11.0, wall=38.5, workers=W_ev, evictions=1, retries=80),   # pod killed -> evicted
               status(12, 21.0, wall=49.0, workers=W_init, evictions=1, retries=80),  # 1-host step, rejoin in flight
               status(13, 26.0, wall=53.5, workers=W_init, evictions=1, retries=80),  # 0.49 but join in flight
               status(14, 26.0, wall=53.5, workers=W_pend, evictions=1, retries=80),
               status(14, 26.0, wall=53.5, workers=W2b, evictions=1, retries=80),    # ACTIVE again (inc 2)
               status(15, 18.0, wall=45.0, workers=W2b, evictions=1, retries=80),
               status(16, 11.0, wall=38.5, workers=W2b, evictions=1, retries=80),
               status(17, 11.0, wall=38.5, workers=W2b, evictions=1, retries=80)]
        kube = FakeKube()
        act = JobSetActuator(kube, templates(), {"tc-roll-0"})
        ctl = TunixController(BandPolicy(cfg), ScriptedClient(seq), act, GainGuard(2, 0.10), DecisionLog(""))
        recs = [ctl.run_once(now=float(k) * 50) for k in range(len(seq))]
        self.assertEqual(kube.applied, [])
        self.assertEqual(kube.deleted, [])
        ext = [r["membership_changed_externally"] for r in recs if "membership_changed_externally" in r]
        self.assertEqual(len(ext), 2)  # evicted, then rejoined
        held = [r["decision"]["reason"] for r in recs if "decision" in r and r["decision"]["inputs"].get("join_in_flight")]
        self.assertTrue(held and all("scaling in progress" in h or "no new param version" in h for h in held))

    def test_scales_out_if_replacement_never_comes_back(self):
        cfg = PolicyConfig(min_standalone=1, max_standalone=3, idle_hi=0.40, idle_lo=0.15, dwell_steps=2,
                           min_action_interval_s=0, mq_hi_frac=0.0)
        W_ev = {"tc-roll-0": ("active", 1, 128), "tc-roll-1": ("evicted", 1, 0)}
        seq = [status(11, 11.0, wall=38.5, workers={"tc-roll-0": ("active", 1, 64), "tc-roll-1": ("active", 1, 64)}),
               status(11, 11.0, wall=38.5, workers=W_ev, evictions=1),
               status(12, 21.0, wall=49.0, workers=W_ev, evictions=1),
               status(13, 26.0, wall=53.5, workers=W_ev, evictions=1),
               status(14, 26.0, wall=53.5, workers=W_ev, evictions=1)]
        kube = FakeKube()
        ctl = TunixController(BandPolicy(cfg), ScriptedClient(seq), JobSetActuator(kube, templates(), {"tc-roll-0"}), None, DecisionLog(""))
        for k in range(len(seq)):
            ctl.run_once(now=float(k) * 50)
        # roll-1 is EVICTED (not live) so its template is free again: the controller re-applies roll-1 first
        self.assertEqual(kube.applied, ["kind: JobSet\nname: tc-roll-1\n"])


class GuardSkipsStepZero(unittest.TestCase):
    def test_before_window_excludes_compile_step(self):
        g = GainGuard(window=2, min_gain=0.10)
        hist = [{"step": 0, "exposed_generation_time": 295.6}, {"step": 1, "exposed_generation_time": 49.8}]
        g.on_scale_out_issued(status(1, 49.8, history=hist), "te-roll-1")
        self.assertEqual(g.pending["before_steps"], [1])
        self.assertAlmostEqual(g.pending["before"], 49.8)


# ------------------------------------------------------------------------------------------------ slot gate
class SlotTracker(unittest.TestCase):
    """Per-step pool slot utilization from /status polls (sum inflight / sum cap over ACTIVE workers)."""

    def test_instant_counts_active_workers_only_and_uses_default_cap(self):
        tr = SlotUtilTracker(default_cap=64)
        st = status(1, 10.0, workers={"te-roll-0": ("active", 1, 32), "te-roll-1": ("initializing", 1, 0), "te-roll-2": ("evicted", 1, 5)}, cap=64)
        x = tr.instant(st, 0.0)
        self.assertEqual((x["active"], x["inflight"], x["cap"]), (1, 32, 64))
        self.assertAlmostEqual(x["util"], 0.5)
        # a worker without an advertised cap takes default_cap
        d = {"t": 1.0, "last_completed_step": 1, "step": {}, "workers": {"a": {"state": "active", "incarnation": 1, "inflight": 10, "max_concurrency": None}}}
        y = tr.instant(TunixStatus.from_json(d), 1.0)
        self.assertEqual(y["cap"], 64)
        self.assertAlmostEqual(y["util"], 10 / 64)
        self.assertIsNone(SlotUtilTracker().instant(TunixStatus.from_json(d), 1.0)["util"])

    def test_samples_belong_to_the_step_in_flight_and_finalize_with_peak_and_time_weighted_mean(self):
        tr = SlotUtilTracker(stat="peak", window=2)
        W = lambda n: {"te-roll-0": ("active", 1, n)}
        # step 1 in flight (last completed 0): 32 in flight for 40 s, then 4 for 20 s, then 0 for 10 s
        self.assertEqual(tr.observe(status(0, 10.0, workers=W(32), cap=64, t=100.0), 100.0), [])
        tr.observe(status(0, 10.0, workers=W(32), cap=64, t=110.0), 110.0)
        tr.observe(status(0, 10.0, workers=W(32), cap=64, t=120.0), 120.0)
        tr.observe(status(0, 10.0, workers=W(32), cap=64, t=130.0), 130.0)
        tr.observe(status(0, 10.0, workers=W(4), cap=64, t=140.0), 140.0)
        tr.observe(status(0, 10.0, workers=W(4), cap=64, t=150.0), 150.0)
        tr.observe(status(0, 10.0, workers=W(0), cap=64, t=160.0), 160.0)
        fin = tr.observe(status(1, 10.0, workers=W(32), cap=64, t=170.0), 170.0)  # step 1 completed -> finalized
        self.assertEqual(len(fin), 1)
        f = fin[0]
        self.assertEqual(f["step"], 1)
        self.assertEqual(f["n_samples"], 7)
        self.assertAlmostEqual(f["peak"], 0.5)
        self.assertEqual((f["inflight_peak"], f["cap"]), (32, 64))
        # hold-last weights: 0.5 x 40 s + 0.0625 x 20 s + 0 x 10 s over 70 s
        self.assertAlmostEqual(f["mean"], (0.5 * 40 + (4 / 64) * 20) / 70, places=6)
        self.assertEqual(tr.window_steps(1), [1])
        self.assertAlmostEqual(tr.window_value(1), 0.5)
        # step 2: 64 in flight (demand == cap) -> peak 1.0; window over steps 1-2 = 0.75
        tr.observe(status(1, 10.0, workers=W(64), cap=64, t=180.0), 180.0)
        tr.observe(status(2, 10.0, workers=W(0), cap=64, t=240.0), 240.0)
        self.assertEqual(tr.window_steps(2), [1, 2])
        self.assertAlmostEqual(tr.window_value(2), 0.75)
        self.assertAlmostEqual(SlotUtilTracker(stat="mean", window=1).window_value(None) or 0.0, 0.0)

    def test_gap_in_steps_finalizes_empty_records(self):
        tr = SlotUtilTracker()
        tr.observe(status(0, 1.0, workers={"a": ("active", 1, 8)}, cap=8, t=0.0), 0.0)
        fin = tr.observe(status(3, 1.0, workers={"a": ("active", 1, 8)}, cap=8, t=50.0), 50.0)
        self.assertEqual([f["step"] for f in fin], [1, 2, 3])
        self.assertEqual(fin[0]["n_samples"], 1)
        self.assertEqual(fin[1]["n_samples"], 0)
        self.assertIsNone(fin[1]["peak"])


class ControllerSlotGate(unittest.TestCase):
    """The te recipe with the gate on: band says scale out, slot utilization says the slots are half empty."""

    def _cfg(self, **kw):
        base = dict(min_standalone=1, max_standalone=3, idle_hi=0.40, idle_lo=0.15, dwell_steps=2,
                    min_action_interval_s=120, mq_lo_frac=0.8, mq_hi_frac=0.0, util_hi=0.75, util_lo=0.40)
        base.update(kw)
        return PolicyConfig(**base)

    def _run(self, seq, cfg, templates_=None, stat="peak"):
        kube, clk = FakeKube(), Clock(0.0)
        act = JobSetActuator(kube, templates_ or templates(), {"tc-roll-0"}, clock=clk)
        tr = SlotUtilTracker(stat=stat, window=cfg.dwell_steps)
        ctl = TunixController(BandPolicy(cfg), ScriptedClient(seq), act, GainGuard(2, 0.10), DecisionLog(""), tracker=tr)
        recs = []
        for k, st in enumerate(seq):
            clk.t = float(st.t)
            recs.append(ctl.run_once(now=clk.t))
        return kube, recs, ctl

    def test_te_demand_32_of_64_blocks_scale_out_and_records_band_request(self):
        W = lambda n: {"tc-roll-0": ("active", 1, n)}
        seq = []
        t = 0.0
        # steps 1..6 at the te cadence (last completed = step - 1 while step is in flight): 32 in flight for most of
        # the step, draining to a few; wait share 0.79 at every completed step
        for step in range(1, 7):
            for infl in (32, 32, 32, 32, 6, 1):
                seq.append(status(step - 1, 49.8, wall=62.7, workers=W(infl), cap=64, t=t))
                t += 10.0
        seq.append(status(6, 49.8, wall=62.7, workers=W(32), cap=64, t=t))
        kube, recs, ctl = self._run(seq, self._cfg())
        self.assertEqual(kube.applied, [])
        self.assertEqual(kube.deleted, [])
        gates = [r["gate"] for r in recs if "gate" in r]
        self.assertTrue(gates)
        # band-only shadow asked for a scale-out, the gate refused: utilization window 0.5 < 0.75
        blocked = [g for g in gates if g["verdict"] == "blocked_by_util"]
        self.assertTrue(blocked, [g["verdict"] for g in gates])
        self.assertEqual(blocked[0]["band"], "scale_out")
        self.assertGreaterEqual(blocked[0]["band_hi_count"], 2)
        self.assertAlmostEqual(blocked[0]["slot_util_window"], 0.5)
        self.assertEqual(blocked[0]["policy_hi_count"], 0)
        # the policy's own inputs carry the slot utilization and the per-step gate view
        last = [r for r in recs if "decision" in r][-1]["decision"]["inputs"]
        self.assertAlmostEqual(last["slot_util"], 0.5)
        self.assertIn("slot_gate", last)
        self.assertEqual(last["slot_gate"]["per_step"][max(last["slot_gate"]["per_step"])]["inflight_peak"], 32)
        # step_util records were emitted when steps completed
        su = [r["step_util"] for r in recs if "step_util" in r]
        self.assertTrue(all(x["peak"] == 0.5 for x in su if x["n_samples"]))
        # 1 host at min: the util < util_lo scale-in path cannot fire (0.5 >= 0.40 anyway) and min holds
        self.assertTrue(all(r["decision"]["action"] == "hold" for r in recs if "decision" in r))

    def test_d1_demand_64_of_64_allows_scale_out_then_gate_holds_at_two(self):
        W1 = lambda n: {"tc-roll-0": ("active", 1, n)}
        W1j = lambda n: {"tc-roll-0": ("active", 1, n), "tc-roll-1": ("initializing", 1, 0)}
        W2 = lambda a, b: {"tc-roll-0": ("active", 1, a), "tc-roll-1": ("active", 1, b)}
        seq = []
        t = 0.0
        def step_polls(last, wait, wall, mk, levels):
            nonlocal t
            for n in levels:
                seq.append(status(last, wait, wall=wall, workers=mk(n), cap=64, t=t))
                t += 10.0
        step_polls(0, 43.1, wall=127.3, mk=W1, levels=(64, 64, 64, 64, 64, 30, 6))     # step 1 in flight; step 0 share 0.34
        step_polls(1, 101.2, wall=125.6, mk=W1, levels=(64, 64, 64, 64, 64, 30, 6))    # step 2 in flight; share 0.81 (hi 1)
        kube, recs, ctl = self._run(seq + [status(2, 101.2, wall=125.6, workers=W1(64), cap=64, t=t)], self._cfg())
        # share > 0.40 at steps 1 and 2, window peak 1.0 >= 0.75 -> scale_out allowed at step 2
        self.assertEqual(kube.applied, ["kind: JobSet\nname: tc-roll-1\n"])
        dec = [r for r in recs if r.get("decision", {}).get("action") == "scale_out"]
        self.assertEqual(len(dec), 1)
        self.assertEqual(dec[0]["gate"]["verdict"], "allowed_by_util")
        self.assertAlmostEqual(dec[0]["decision"]["inputs"]["slot_util"], 1.0)
        self.assertIn("slot_util>=0.75", dec[0]["decision"]["reason"])

    def test_two_hosts_with_half_empty_slots_scale_in_by_util_and_min_one_holds(self):
        W2 = lambda a, b: {"tc-roll-0": ("active", 1, a), "tc-roll-1": ("active", 1, b)}
        cfg = self._cfg(min_action_interval_s=0)
        seq = []
        t = 0.0
        for last in (9, 10, 11):
            for n in (16, 16, 16, 2):
                seq.append(status(last, 50.0, wall=65.0, workers=W2(n, n), cap=64, t=t))  # 32 / 128 = 0.25 < 0.40
                t += 10.0
        seq.append(status(12, 50.0, wall=65.0, workers=W2(16, 16), cap=64, t=t))
        kube, recs, ctl = self._run(seq, cfg)
        self.assertEqual(kube.deleted, ["tc-roll-1"])
        d = [r for r in recs if r.get("decision", {}).get("action") == "scale_in"][0]
        self.assertEqual(d["gate"]["verdict"], "scale_in_by_util")
        self.assertIn("slot_util<0.4", d["decision"]["reason"])
        # same utilization at 1 host: proposal is held at min_standalone
        W1 = lambda n: {"tc-roll-0": ("active", 1, n)}
        seq1 = [status(s, 50.0, wall=65.0, workers=W1(16), cap=64, t=float(k) * 10) for k, s in enumerate((9, 9, 10, 10, 11, 11, 12))]
        kube1, recs1, _ = self._run(seq1, cfg)
        self.assertEqual(kube1.deleted, [])
        reasons = [r["decision"]["reason"] for r in recs1 if "decision" in r]
        self.assertTrue(any("at min_standalone" in x for x in reasons), reasons)

    def test_join_in_flight_holds_even_when_util_says_scale_in(self):
        Wj = {"tc-roll-0": ("active", 1, 8), "tc-roll-1": ("initializing", 1, 0), "tc-roll-2": ("active", 1, 8)}
        cfg = self._cfg(min_action_interval_s=0)
        seq = [status(s, 50.0, wall=65.0, workers=Wj, cap=64, t=float(k) * 10) for k, s in enumerate((5, 5, 6, 6, 7, 7, 8))]
        kube, recs, _ = self._run(seq, cfg)
        self.assertEqual(kube.deleted, [])
        self.assertTrue(all(r["decision"]["reason"] in ("scaling in progress", "no new param version") for r in recs if "decision" in r))

    def test_gate_off_keeps_te_behaviour(self):
        W = lambda n: {"tc-roll-0": ("active", 1, n)}
        cfg = PolicyConfig(min_standalone=1, max_standalone=3, idle_hi=0.40, idle_lo=0.15, dwell_steps=2,
                           min_action_interval_s=0, mq_hi_frac=0.0)
        seq = [status(s, 49.8, wall=62.7, workers=W(32), cap=64, t=float(k) * 10) for k, s in enumerate((0, 1, 1, 2))]
        kube = FakeKube()
        ctl = TunixController(BandPolicy(cfg), ScriptedClient(seq), JobSetActuator(kube, templates(), {"tc-roll-0"}), None, DecisionLog(""))
        recs = [ctl.run_once(now=float(k) * 10) for k in range(len(seq))]
        self.assertEqual(len(kube.applied), 1)
        self.assertTrue(all("gate" not in r for r in recs))


class SlotGateMixedPoolStep(unittest.TestCase):
    """Seen live (tg2): the step during which the second host was promoted peaked at 64/64 on the OLD pool; with that
    step in the window the mean (1.0 + 0.5) / 2 = 0.75 passed the gate and a third host was added. Steps that mix two
    pool sizes, or were measured at another pool size than the current one, must not enter the window."""

    def test_mixed_pool_step_is_excluded_from_the_window(self):
        tr = SlotUtilTracker(stat="peak", window=2)
        W1 = lambda n: {"tc-roll-0": ("active", 1, n)}
        W2 = lambda a, b: {"tc-roll-0": ("active", 1, a), "tc-roll-1": ("active", 1, b)}
        t = 0.0
        for n in (64, 64, 30):                                   # step 3 in flight at 1 host: 64/64
            tr.observe(status(2, 80.0, workers=W1(n), cap=64, t=t), t); t += 10
        tr.observe(status(2, 80.0, workers=W2(5, 0), cap=64, t=t), t); t += 10   # second host promoted mid-step
        for n in (32, 32, 6):                                    # step 4 in flight at 2 hosts: 64/128
            tr.observe(status(3, 60.0, workers=W2(n, n), cap=64, t=t), t); t += 10
        tr.observe(status(4, 60.0, workers=W2(32, 32), cap=64, t=t), t)
        self.assertEqual(tr.per_step[3]["active_min"], 1)
        self.assertEqual(tr.per_step[3]["active_max"], 2)
        self.assertAlmostEqual(tr.per_step[3]["peak"], 1.0)
        self.assertAlmostEqual(tr.per_step[4]["peak"], 0.5)
        # naive window would be 0.75; the mixed step 3 is excluded
        self.assertEqual(tr.window_steps(4, active=2), [4])
        self.assertAlmostEqual(tr.window_value(4, active=2), 0.5)
        self.assertEqual(tr.gate_view(4, active=2)["excluded_mixed_pool_steps"], [3])
        # a clean step measured at ANOTHER pool size than the current one is excluded too
        self.assertEqual(tr.window_steps(4, active=1), [])

    def test_controller_does_not_add_third_host_on_the_join_step(self):
        cfg = PolicyConfig(min_standalone=1, max_standalone=3, idle_hi=0.40, idle_lo=0.15, dwell_steps=2,
                           min_action_interval_s=0, mq_hi_frac=0.0, util_hi=0.75, util_lo=0.40)
        W1 = lambda n: {"tc-roll-0": ("active", 1, n)}
        W1j = lambda n: {"tc-roll-0": ("active", 1, n), "tc-roll-1": ("initializing", 1, 0)}
        W2 = lambda a, b: {"tc-roll-0": ("active", 1, a), "tc-roll-1": ("active", 1, b)}
        seq = []; t = [0.0]
        def polls(last, wait, wall, mk, levels, sip=None):
            for n in levels:
                seq.append(status(last, wait, wall=wall, workers=mk(n), cap=64, t=t[0], sip=sip)); t[0] += 10
        polls(0, 43.0, 127.0, W1, (64, 64, 64, 20))        # step 1 in flight at 1 host
        polls(1, 80.0, 105.0, W1, (64, 64, 64, 20))        # step 2 in flight; share 0.76 x2 -> scale_out at step 2 (window 1.0)
        polls(2, 80.0, 105.0, W1, (64,))                   # step 2 completes -> scale_out decision on this poll
        polls(2, 80.0, 105.0, W1j, (64, 64))               # step 3 in flight, join in progress
        polls(2, 80.0, 105.0, lambda n: W2(n, 0), (20,), sip=3)   # promoted mid-step 3
        W2s = lambda n: W2(n, n)
        polls(3, 88.0, 113.0, W2s, (32, 32, 32, 6))        # step 4 at 2 hosts: 64/128 = 0.5
        polls(4, 31.0, 58.0, W2s, (32, 32, 6))             # step 5
        polls(5, 31.0, 58.0, W2s, (32, 32, 6))
        seq.append(status(6, 31.0, wall=58.0, workers=W2(32, 32), cap=64, t=t[0]))
        kube, clk = FakeKube(), Clock(0.0)
        act = JobSetActuator(kube, templates(), {"tc-roll-0"}, clock=clk)
        ctl = TunixController(BandPolicy(cfg), ScriptedClient(seq), act, GainGuard(2, 0.10), DecisionLog(""), tracker=SlotUtilTracker("peak", 2))
        recs = []
        for st in seq:
            clk.t = float(st.t); recs.append(ctl.run_once(now=clk.t))
        self.assertEqual(kube.applied, ["kind: JobSet\nname: tc-roll-1\n"])   # one join only
        outs = [r for r in recs if r.get("decision", {}).get("action") == "scale_out"]
        self.assertEqual(len(outs), 1)
        # after the join the window never reaches 0.75 again: step 3 (mixed) excluded, steps 4+ at 0.5
        later = [r["decision"]["inputs"]["slot_util"] for r in recs if "decision" in r and r["decision"]["inputs"]["param_version"] >= 4 and r["decision"]["inputs"].get("slot_util") is not None]
        self.assertTrue(later and max(later) <= 0.5, later)


class QueueTerm(unittest.TestCase):
    """Coordinator design change after tg11 step 11: slots full AND pending queue empty = stuck on sandboxes, not a
    reason to scale out. With QUEUE_MIN set, a scale-out decided by band + util is actuated only if the window mean
    of the orchestrator pending queue is >= QUEUE_MIN."""

    def _run(self, pending, queue_min=8.0):
        cfg = PolicyConfig(min_standalone=1, max_standalone=3, idle_hi=0.15, idle_lo=0.0, dwell_steps=2,
                           min_action_interval_s=0, mq_hi_frac=0.0, util_hi=0.75, util_lo=0.40)
        W = lambda n: {"tc-roll-0": ("active", 1, n)}
        seq = []; t = 0.0
        for last in (0, 1, 2):
            for n in (128, 128, 128, 128):
                seq.append(status(last, 60.0, wall=230.0, workers=W(n), cap=128, pending=pending, t=t)); t += 10
        seq.append(status(3, 60.0, wall=230.0, workers=W(128), cap=128, pending=pending, t=t))
        kube, clk = FakeKube(), Clock(0.0)
        act = JobSetActuator(kube, templates(), {"tc-roll-0"}, clock=clk)
        ctl = TunixController(BandPolicy(cfg), ScriptedClient(seq), act, GainGuard(2, 0.10), DecisionLog(""),
                              tracker=SlotUtilTracker("mean", 2), queue_min=queue_min)
        recs = []
        for st in seq:
            clk.t = float(st.t); recs.append(ctl.run_once(now=clk.t))
        return kube, recs

    def test_full_slots_empty_queue_is_blocked(self):
        kube, recs = self._run(pending=0)
        self.assertEqual(kube.applied, [])
        blocked = [r for r in recs if r.get("actuation", {}).get("result") == "blocked_by_queue"]
        self.assertTrue(blocked)
        self.assertEqual(blocked[0]["gate"]["verdict"], "blocked_by_queue")
        self.assertAlmostEqual(blocked[0]["gate"]["queue_window"], 0.0)
        self.assertAlmostEqual(blocked[0]["decision"]["inputs"]["slot_util"], 1.0)

    def test_full_slots_deep_queue_is_allowed(self):
        kube, recs = self._run(pending=200)
        self.assertEqual(kube.applied, ["kind: JobSet\nname: tc-roll-1\n"])
        allowed = [r for r in recs if r.get("decision", {}).get("action") == "scale_out"]
        self.assertEqual(allowed[0]["gate"]["verdict"], "allowed_by_util")
        self.assertGreaterEqual(allowed[0]["gate"]["queue_window"], 8.0)

    def test_per_step_pending_mean_is_time_weighted(self):
        tr = SlotUtilTracker("mean", 2)
        W = {"tc-roll-0": ("active", 1, 128)}
        tr.observe(status(0, 1.0, workers=W, cap=128, pending=300, t=0.0), 0.0)
        tr.observe(status(0, 1.0, workers=W, cap=128, pending=100, t=30.0), 30.0)
        tr.observe(status(1, 1.0, workers=W, cap=128, pending=0, t=40.0), 40.0)
        self.assertAlmostEqual(tr.per_step[1]["pending_mean"], (300 * 30 + 100 * 10) / 40)
        self.assertEqual(tr.per_step[1]["pending_peak"], 300)
        self.assertAlmostEqual(tr.queue_window_value(1, active=1), (300 * 30 + 100 * 10) / 40)
