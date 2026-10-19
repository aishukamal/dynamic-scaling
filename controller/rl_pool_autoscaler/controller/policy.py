"""BandPolicy: target band on trainer idle ratio with dwell, queue guards and action spacing.

Pure Python; the controller loop feeds it one ``Observation`` per poll. Decisions are made only on
NEW samples (a new ``param_version``), because the trainer-side numbers (wait share per sync cycle)
only change once per parameter sync; polling more often than that would otherwise inflate dwell
counts. The policy therefore counts dwell in *param versions*, not in polls.

Rules (defaults in ``PolicyConfig``):
    scale out by 1   idle_ratio > idle_hi for ``dwell_steps`` consecutive versions
                     AND mq_size < mq_lo_frac * max_queue_size        (queue not already full)
                     AND active < max
    scale in by 1    idle_ratio < idle_lo for ``dwell_steps`` consecutive versions
                     AND mq_size >= mq_hi_frac * max_queue_size       (samplers over-producing)
                     AND active > min
    hold             otherwise; also while a scale is in progress and for ``min_action_interval_s``
                     after the last action (the dwell counters are frozen, not reset, during that
                     spacing window, so a sustained condition still fires as soon as spacing allows)

``idle_ratio`` here is the trainer's wait share: ``trainer_wait_s / trainer_step_s`` for the last
sync cycle, exported by the control actor as ``trainer_idle_ratio``.

Sampler slot-utilization gate (optional, ``util_hi`` / ``util_lo``; GPU-M2 addition):
    The M0 ladder measured the SAME trainer idle ratio (0.94) at 1, 2 and 4 samplers, so the idle
    band alone cannot find the knee of an agentic workload whose step time is bounded by tool
    latency. The signal that does move with the pool size is the pool slot utilization
        slot_util = rollout_resource_utilization * max_concurrent_samples / (active * cspr)
    i.e. the time-weighted mean number of in-flight trajectories over the version window divided
    by the admission capacity the active replicas offer (``cspr`` = concurrent_samples_per_replica).
    With the gate on: scale out additionally requires slot_util >= util_hi (adding a replica only
    helps while the existing slots are busy); scale in ALSO fires when slot_util < util_lo for
    ``dwell_steps`` versions (the pool has idle slots even though the trainer queue is not full).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PolicyConfig:
    min_standalone: int = 1
    max_standalone: int = 8
    idle_hi: float = 0.50
    idle_lo: float = 0.15
    dwell_steps: int = 2
    min_action_interval_s: float = 600.0
    mq_lo_frac: float = 0.8
    mq_hi_frac: float = 0.9
    util_hi: float | None = None  # None disables the slot-utilization gate
    util_lo: float | None = None
    # Floor guard (DECISIONS.md 13): do not scale out again if the last join did not reduce the
    # trainer wait by at least floor_min_gain (fraction of the pre-join wait, means over dwell
    # versions). None disables. The block clears when the wait rises back above the post-join
    # level by the same margin (conditions changed, e.g. drift).
    floor_min_gain: float | None = None
    # Versions whose wait is NOT used by the floor guard (start-up transient: the first version of every
    # measured run was 25-50 % shorter than steady state, which made the 2-version baseline read "no gain"
    # after the first join in both live runs). Waits of param_version <= floor_skip_versions are ignored.
    floor_skip_versions: int = 0

    def validate(self) -> None:
        if self.min_standalone < 1 or self.max_standalone < self.min_standalone:
            raise ValueError("need 1 <= min_standalone <= max_standalone")
        if not (0.0 <= self.idle_lo < self.idle_hi <= 1.0):
            raise ValueError("need 0 <= idle_lo < idle_hi <= 1")
        if self.dwell_steps < 1:
            raise ValueError("dwell_steps >= 1")
        if (self.util_hi is None) != (self.util_lo is None):
            raise ValueError("util_hi and util_lo must be set together")
        if self.util_hi is not None and not (0.0 <= self.util_lo < self.util_hi <= 1.0):
            raise ValueError("need 0 <= util_lo < util_hi <= 1")


@dataclass
class Observation:
    t: float
    param_version: int
    trainer_idle_ratio: float
    mq_size: float
    max_queue_size: float
    active_standalone: int
    desired_standalone: int | None = None
    scaling_in_progress: bool = False
    rollout_util: float | None = None  # dynamic_resource/rollout_resource_utilization (last version)
    max_concurrent_samples: float | None = None
    concurrent_samples_per_replica: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def slot_util(self) -> float | None:
        """Mean in-flight trajectories / (active replicas x per-replica admission cap)."""
        if self.rollout_util is None or not self.max_concurrent_samples or not self.concurrent_samples_per_replica:
            return None
        if self.active_standalone <= 0:
            return None
        return float(self.rollout_util) * float(self.max_concurrent_samples) / (self.active_standalone * float(self.concurrent_samples_per_replica))


@dataclass
class Decision:
    t: float
    action: str  # "scale_out" | "scale_in" | "hold"
    desired: int
    reason: str
    inputs: dict[str, Any]
    counters: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BandPolicy:
    def __init__(self, cfg: PolicyConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.hi_count = 0
        self.lo_count = 0
        self.last_version: int | None = None
        self.last_action_t: float | None = None
        self.last_action: str | None = None
        self._lo_reason: str = "over-provisioned"
        self.decisions_total = {"scale_out": 0, "scale_in": 0, "hold": 0}
        # floor guard state
        self._wait_hist: list[float] = []  # trainer wait per NEW version (most recent last)
        self._pending_join: dict | None = None  # {"baseline": float, "post": [floats]} after a scale_out
        self.floor_blocked: dict | None = None  # {"baseline", "post", "gain"} while blocked

    def observe(self, o: Observation) -> Decision:
        c = self.cfg
        slot_util = o.slot_util()
        inputs = {
            "param_version": o.param_version,
            "trainer_idle_ratio": o.trainer_idle_ratio,
            "mq_size": o.mq_size,
            "max_queue_size": o.max_queue_size,
            "active": o.active_standalone,
            "desired": o.desired_standalone,
            "scaling_in_progress": o.scaling_in_progress,
            "rollout_util": o.rollout_util,
            "max_concurrent_samples": o.max_concurrent_samples,
            "concurrent_samples_per_replica": o.concurrent_samples_per_replica,
            "slot_util": slot_util,
            **o.extra,
        }
        active = int(o.active_standalone)
        gate_on = c.util_hi is not None

        new_sample = self.last_version is None or o.param_version != self.last_version
        if new_sample:
            self.last_version = o.param_version
            self._floor_update(o)
            mq_frac = (o.mq_size / o.max_queue_size) if o.max_queue_size > 0 else 0.0
            busy_ok = (not gate_on) or (slot_util is not None and slot_util >= c.util_hi)
            if o.trainer_idle_ratio > c.idle_hi and mq_frac < c.mq_lo_frac and busy_ok:
                self.hi_count += 1
            else:
                self.hi_count = 0
            # Scale-in (coordinator rule, 2026-10-17, applied before the verl document series): the trainer wait band is
            # primary and the slot-utilisation term is a gate, so BOTH must hold for the dwell: trainer idle share <= IDLE_LO
            # AND (gate on) slot utilisation < UTIL_LO. The previous OR ("queue full and idle < IDLE_LO" or slot-only) let a
            # slot-only reading scale in while the trainer still waited 40 % of the step (tpum3-dynr2 v9).
            idle_low = o.trainer_idle_ratio <= c.idle_lo
            slots_idle = (not gate_on) or (slot_util is not None and slot_util < c.util_lo)
            if idle_low and slots_idle:
                self.lo_count += 1
                self._lo_reason = "idle<=%.2f and slot_util<%.2f" % (c.idle_lo, c.util_lo) if gate_on else "idle<=%.2f" % c.idle_lo
            else:
                self.lo_count = 0

        counters = {"hi_count": self.hi_count, "lo_count": self.lo_count, "new_sample": new_sample, "floor_blocked": self.floor_blocked}

        if not new_sample:
            return self._hold(o, active, "no new param version", inputs, counters)
        if o.scaling_in_progress:
            return self._hold(o, active, "scaling in progress", inputs, counters)
        if o.desired_standalone is not None and o.desired_standalone != active:
            return self._hold(o, active, "pool not yet converged to desired", inputs, counters)
        if self.last_action_t is not None and (o.t - self.last_action_t) < c.min_action_interval_s:
            return self._hold(o, active, "min_action_interval_s not elapsed", inputs, counters)

        if self.hi_count >= c.dwell_steps:
            if active >= c.max_standalone:
                return self._hold(o, active, "sampler-bound but at max_standalone", inputs, counters)
            if self.floor_blocked is not None:
                return self._hold(o, active, "floor guard: last join did not reduce the trainer wait", inputs, counters)
            if self._pending_join is not None:
                return self._hold(o, active, "floor guard: still measuring the last join", inputs, counters)
            why = f"idle>{c.idle_hi} for {self.hi_count} versions and queue not full" + (f" and slot_util>={c.util_hi}" if gate_on else "")
            return self._act(o, "scale_out", active + 1, why, inputs, counters)
        if self.lo_count >= c.dwell_steps:
            if active <= c.min_standalone:
                return self._hold(o, active, "over-provisioned but at min_standalone", inputs, counters)
            return self._act(o, "scale_in", active - 1, f"{getattr(self, '_lo_reason', 'over-provisioned')} for {self.lo_count} versions", inputs, counters)
        return self._hold(o, active, "within band / dwell not met", inputs, counters)

    # ------------------------------------------------------------------ floor guard
    def _floor_update(self, o: Observation) -> None:
        c = self.cfg
        if c.floor_min_gain is None:
            return
        w = o.extra.get("trainer_wait_s") if isinstance(o.extra, dict) else None
        if w is None:
            return
        if o.param_version <= c.floor_skip_versions:
            return  # start-up transient: not evidence for or against a join
        w = float(w)
        self._wait_hist.append(w)
        if len(self._wait_hist) > 64:
            del self._wait_hist[:-64]
        pj = self._pending_join
        if pj is not None:
            # count post-join versions only once the pool has converged to the requested size
            if o.desired_standalone is None or o.active_standalone >= pj["target"]:
                pj["post"].append(w)
            if len(pj["post"]) >= c.dwell_steps:
                post = sum(pj["post"]) / len(pj["post"])
                gain = (pj["baseline"] - post) / pj["baseline"] if pj["baseline"] > 0 else 0.0
                if gain < c.floor_min_gain:
                    self.floor_blocked = {"baseline": pj["baseline"], "post": post, "gain": gain, "version": o.param_version}
                self._pending_join = None
        elif self.floor_blocked is not None:
            # clear when the wait has risen back above the post-join level by the margin (conditions changed)
            recent = self._wait_hist[-c.dwell_steps:]
            if len(recent) >= c.dwell_steps and (sum(recent) / len(recent)) > self.floor_blocked["post"] * (1.0 + c.floor_min_gain):
                self.floor_blocked = None

    def _floor_arm(self, action: str, desired: int) -> None:
        c = self.cfg
        if c.floor_min_gain is None:
            return
        if action == "scale_out":
            recent = self._wait_hist[-c.dwell_steps:]
            if len(recent) >= c.dwell_steps:  # no baseline yet (start-up) -> the join is not judged
                self._pending_join = {"baseline": sum(recent) / len(recent), "post": [], "target": desired}
        else:
            self._pending_join = None
            self.floor_blocked = None  # a scale-in resets the floor evidence

    def _act(self, o: Observation, action: str, desired: int, reason: str, inputs: dict, counters: dict) -> Decision:
        self.last_action_t = o.t
        self.last_action = action
        self._floor_arm(action, desired)
        self.hi_count = 0
        self.lo_count = 0
        self.decisions_total[action] += 1
        return Decision(t=o.t, action=action, desired=desired, reason=reason, inputs=inputs, counters=dict(counters, reset=True))

    def _hold(self, o: Observation, active: int, reason: str, inputs: dict, counters: dict) -> Decision:
        self.decisions_total["hold"] += 1
        desired = o.desired_standalone if o.desired_standalone is not None else active
        return Decision(t=o.t, action="hold", desired=desired, reason=reason, inputs=inputs, counters=counters)
