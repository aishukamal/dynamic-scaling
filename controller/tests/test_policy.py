import unittest

from rl_pool_autoscaler.controller.policy import BandPolicy, Observation, PolicyConfig


def cfg(**kw):
    base = dict(min_standalone=1, max_standalone=6, idle_hi=0.5, idle_lo=0.15, dwell_steps=2, min_action_interval_s=600, mq_lo_frac=0.8, mq_hi_frac=0.9)
    base.update(kw)
    return PolicyConfig(**base)


class Sim:
    """Feed a synthetic trace: each entry = (idle_ratio, mq_frac); one new param version per entry
    (plus optional repeated polls of the same version). Applies decisions to the fake pool so the
    policy sees its own effect (desired == active after a 'converge' delay of 1 version)."""

    def __init__(self, policy: BandPolicy, active: int = 1, max_queue: int = 160, dt: float = 300.0):
        self.p = policy
        self.active = active
        self.desired = active
        self.max_queue = max_queue
        self.t = 0.0
        self.dt = dt
        self.version = 0
        self.decisions = []

    def poll(self, idle, mq_frac, new_version=True, in_progress=False):
        if new_version:
            self.version += 1
            self.t += self.dt
            self.active = self.desired  # previous action converged
        o = Observation(
            t=self.t,
            param_version=self.version,
            trainer_idle_ratio=idle,
            mq_size=mq_frac * self.max_queue,
            max_queue_size=self.max_queue,
            active_standalone=self.active,
            desired_standalone=self.desired,
            scaling_in_progress=in_progress,
        )
        d = self.p.observe(o)
        self.decisions.append(d)
        if d.action != "hold":
            self.desired = d.desired
        return d

    def actions(self):
        return [d.action for d in self.decisions if d.action != "hold"]


class SamplerBoundRise(unittest.TestCase):
    def test_scale_out_after_dwell_only(self):
        s = Sim(BandPolicy(cfg()))
        d1 = s.poll(0.7, 0.1)
        self.assertEqual(d1.action, "hold")  # dwell=2: first high sample does not act
        d2 = s.poll(0.7, 0.1)
        self.assertEqual(d2.action, "scale_out")
        self.assertEqual(d2.desired, 2)
        self.assertIn("idle>0.5", d2.reason)

    def test_steps_up_one_per_action_until_band(self):
        s = Sim(BandPolicy(cfg(min_action_interval_s=0)))
        trace = [0.8, 0.8, 0.7, 0.7, 0.55, 0.55, 0.3, 0.3, 0.3]
        for idle in trace:
            s.poll(idle, 0.1)
        self.assertEqual(s.actions(), ["scale_out", "scale_out", "scale_out"])
        self.assertEqual(s.active, 4)

    def test_queue_full_blocks_scale_out(self):
        s = Sim(BandPolicy(cfg()))
        for _ in range(4):
            d = s.poll(0.9, 0.95)  # idle high but queue already full: trainer is slow, not the samplers
        self.assertEqual(s.actions(), [])
        self.assertIn("within band", d.reason)

    def test_respects_max(self):
        s = Sim(BandPolicy(cfg(max_standalone=2, min_action_interval_s=0)), active=2)
        for _ in range(4):
            d = s.poll(0.9, 0.1)
        self.assertEqual(s.actions(), [])
        self.assertIn("at max_standalone", d.reason)


class ConvergeAndDrift(unittest.TestCase):
    def test_converge_then_hold(self):
        s = Sim(BandPolicy(cfg(min_action_interval_s=0)))
        for idle in [0.8, 0.8, 0.3, 0.3, 0.3, 0.3]:
            s.poll(idle, 0.2)
        self.assertEqual(s.actions(), ["scale_out"])
        self.assertEqual(s.active, 2)

    def test_slow_drift_eventually_acts(self):
        s = Sim(BandPolicy(cfg(min_action_interval_s=0)))
        trace = [0.40, 0.44, 0.48, 0.52, 0.51, 0.53]
        for idle in trace:
            s.poll(idle, 0.1)
        # 0.52 is first above hi; 0.51 second -> act on the 5th sample
        acts = [i for i, d in enumerate(s.decisions) if d.action == "scale_out"]
        self.assertEqual(acts, [4])

    def test_scale_in_when_over_provisioned(self):
        s = Sim(BandPolicy(cfg(min_action_interval_s=0)), active=4)
        for _ in range(2):
            d = s.poll(0.05, 0.95)
        self.assertEqual(d.action, "scale_in")
        self.assertEqual(d.desired, 3)

    def test_scale_in_needs_full_queue(self):
        s = Sim(BandPolicy(cfg(min_action_interval_s=0)), active=4)
        for _ in range(4):
            s.poll(0.05, 0.5)  # low idle but queue half full: trainer-bound and samplers keeping up; hold
        self.assertEqual(s.actions(), [])

    def test_never_below_min(self):
        s = Sim(BandPolicy(cfg(min_standalone=2, min_action_interval_s=0)), active=2)
        for _ in range(4):
            d = s.poll(0.0, 1.0)
        self.assertEqual(s.actions(), [])
        self.assertIn("at min_standalone", d.reason)


class NoThrash(unittest.TestCase):
    def test_alternating_samples_do_not_trigger(self):
        s = Sim(BandPolicy(cfg(min_action_interval_s=0)))
        for idle in [0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7]:
            s.poll(idle, 0.1)
        self.assertEqual(s.actions(), [])

    def test_min_action_interval_spaces_actions(self):
        s = Sim(BandPolicy(cfg(min_action_interval_s=1000)), active=1)
        s.dt = 300.0
        for _ in range(8):
            s.poll(0.9, 0.1)
        # actions at version 2 (t=600) and then only once 1000 s elapsed (t>=1600 -> version 6)
        acts = [(d.t, d.action) for d in s.decisions if d.action != "hold"]
        self.assertEqual([a for _, a in acts], ["scale_out", "scale_out"])
        self.assertGreaterEqual(acts[1][0] - acts[0][0], 1000)

    def test_repeated_polls_of_same_version_do_not_inflate_dwell(self):
        s = Sim(BandPolicy(cfg()))
        s.poll(0.9, 0.1)
        for _ in range(10):
            d = s.poll(0.9, 0.1, new_version=False)
        self.assertEqual(d.action, "hold")
        self.assertEqual(d.reason, "no new param version")
        d = s.poll(0.9, 0.1)
        self.assertEqual(d.action, "scale_out")

    def test_holds_while_scaling_in_progress_or_unconverged(self):
        s = Sim(BandPolicy(cfg(min_action_interval_s=0)))
        s.poll(0.9, 0.1)
        d = s.poll(0.9, 0.1, in_progress=True)
        self.assertEqual(d.action, "hold")
        self.assertIn("in progress", d.reason)
        # desired != active -> hold
        s.desired = 3
        s.active = 1
        d = s.poll(0.9, 0.1)
        s.active = 1  # Sim set active=desired; force unconverged view
        o = Observation(t=s.t, param_version=s.version + 1, trainer_idle_ratio=0.9, mq_size=1, max_queue_size=160, active_standalone=1, desired_standalone=3)
        d = s.p.observe(o)
        self.assertEqual(d.action, "hold")
        self.assertIn("not yet converged", d.reason)

    def test_decision_carries_inputs_for_replay(self):
        s = Sim(BandPolicy(cfg()))
        d = s.poll(0.7, 0.1)
        rec = d.as_dict()
        for k in ("param_version", "trainer_idle_ratio", "mq_size", "max_queue_size", "active", "desired"):
            self.assertIn(k, rec["inputs"])
        self.assertIn("hi_count", rec["counters"])


class ConfigValidation(unittest.TestCase):
    def test_bad_band(self):
        with self.assertRaises(ValueError):
            BandPolicy(cfg(idle_lo=0.6, idle_hi=0.5))
        with self.assertRaises(ValueError):
            BandPolicy(cfg(dwell_steps=0))
        with self.assertRaises(ValueError):
            BandPolicy(cfg(min_standalone=0))


if __name__ == "__main__":
    unittest.main()


class SlotUtilGate(unittest.TestCase):
    """GPU-M2: the sampler slot-utilization gate (util_hi/util_lo) on top of the idle band."""

    def _obs(self, s, idle, util, active, cspr=32, cap=96):
        s.version += 1
        s.t += s.dt
        s.active = s.desired
        o = Observation(t=s.t, param_version=s.version, trainer_idle_ratio=idle, mq_size=1, max_queue_size=cap,
                        active_standalone=s.active, desired_standalone=s.desired, rollout_util=util,
                        max_concurrent_samples=min(s.active * cspr, cap), concurrent_samples_per_replica=cspr)
        d = s.p.observe(o)
        s.decisions.append(d)
        if d.action != "hold":
            s.desired = d.desired
        return d

    def test_slot_util_formula(self):
        o = Observation(t=0, param_version=0, trainer_idle_ratio=0.9, mq_size=0, max_queue_size=96, active_standalone=4,
                        rollout_util=0.6, max_concurrent_samples=96, concurrent_samples_per_replica=32)
        self.assertAlmostEqual(o.slot_util(), 0.6 * 96 / 128)

    def test_ladder_trace_converges_between_knee_arms(self):
        # idle 0.94 at every arm (M0 ladder); rollout util 1.0 (S=1), 0.9 (S=2), 0.62 (S=3), 0.65 (S=4)
        s = Sim(BandPolicy(cfg(util_hi=0.75, util_lo=0.40, idle_hi=0.90, idle_lo=0.50)))
        util = {1: 1.0, 2: 0.9, 3: 0.62, 4: 0.65, 5: 0.62, 6: 0.6}
        for _ in range(14):
            self._obs(s, 0.94, util[s.active], s.active)
        self.assertEqual(s.actions(), ["scale_out", "scale_out"])  # 1 -> 2 -> 3, then hold
        self.assertEqual(s.active, 3)

    def test_scale_in_from_max_when_slots_idle(self):
        s = Sim(BandPolicy(cfg(util_hi=0.75, util_lo=0.40, idle_hi=0.90, idle_lo=0.50, max_standalone=8)), active=8)
        # S=8: util 0.6 of cap 96 over 256 slots = 0.225 -> scale in; S=5: 0.36 -> in; S=4: 0.45 -> hold
        for _ in range(16):
            self._obs(s, 0.94, 0.6, s.active)
        self.assertEqual(s.active, 4)
        self.assertTrue(all(a == "scale_in" for a in s.actions()))

    def test_gate_off_keeps_legacy_behaviour(self):
        s = Sim(BandPolicy(cfg()))
        self._obs(s, 0.7, 0.2, 1)
        d = self._obs(s, 0.7, 0.2, 1)
        self.assertEqual(d.action, "scale_out")

    def test_gate_needs_both(self):
        with self.assertRaises(ValueError):
            BandPolicy(PolicyConfig(util_hi=0.7))


class FloorGuard(unittest.TestCase):
    """DECISIONS.md 13: do not scale out again if the last join did not reduce the trainer wait."""

    def _obs(self, s, idle, wait, active_override=None):
        s.version += 1
        s.t += s.dt
        s.active = s.desired
        o = Observation(t=s.t, param_version=s.version, trainer_idle_ratio=idle, mq_size=1, max_queue_size=96,
                        active_standalone=s.active, desired_standalone=s.desired, extra={"trainer_wait_s": wait})
        d = s.p.observe(o)
        s.decisions.append(d)
        if d.action != "hold":
            s.desired = d.desired
        return d

    def test_blocks_after_a_join_without_gain_and_clears_on_drift(self):
        s = Sim(BandPolicy(cfg(floor_min_gain=0.05, idle_hi=0.5, max_standalone=8)))
        # Sim semantics: the wait reported for a version is the one measured with the PREVIOUS active count
        # (the version in which a join lands is mixed), so a no-gain join is detected one version later.
        waits = {1: 800.0, 2: 650.0, 3: 650.0, 4: 650.0, 5: 650.0}  # 2 -> 3 brings no gain
        for _ in range(12):
            self._obs(s, 0.9, waits[s.active])
        self.assertEqual(s.actions(), ["scale_out", "scale_out", "scale_out"])
        self.assertEqual(s.active, 4)
        self.assertIsNotNone(s.p.floor_blocked)
        self.assertTrue(any("floor guard" in d.reason for d in s.decisions))
        # drift: wait rises well above the post-join level -> block clears -> scale out resumes
        n_before = len(s.actions())
        for _ in range(4):
            self._obs(s, 0.9, 900.0)
        self.assertEqual(len(s.actions()), n_before + 1)  # the block cleared and one more join was made
        self.assertEqual(s.desired, 5)

    def test_disabled_by_default(self):
        s = Sim(BandPolicy(cfg()))
        for _ in range(8):
            self._obs(s, 0.9, 650.0)
        self.assertEqual(s.desired, 5)


class FloorGuardSkipsStartup(unittest.TestCase):
    def test_first_join_not_judged_against_the_transient(self):
        # live pattern: v1 short (transient), v2 long, join at v2, S=2 waits ~ v2 level -> must NOT block
        s = Sim(BandPolicy(cfg(floor_min_gain=0.05, floor_skip_versions=2, idle_hi=0.5, max_standalone=8)))
        fg = FloorGuard()
        waits = [385.0, 639.0, 519.0, 791.0, 598.0, 553.0, 695.0]
        for w in waits:
            fg._obs(s, 0.93, w)
        self.assertFalse(any("floor guard" in d.reason for d in s.decisions))
        self.assertEqual(s.actions(), ["scale_out", "scale_out", "scale_out"])  # v2, v4, v6 (no floor block)

    def test_later_joins_still_judged(self):
        s = Sim(BandPolicy(cfg(floor_min_gain=0.05, floor_skip_versions=2, idle_hi=0.5, max_standalone=8)))
        fg = FloorGuard()
        for w in [385.0, 639.0, 650.0, 650.0, 650.0, 650.0, 650.0, 650.0, 650.0]:
            fg._obs(s, 0.93, w)
        self.assertIsNotNone(s.p.floor_blocked)
