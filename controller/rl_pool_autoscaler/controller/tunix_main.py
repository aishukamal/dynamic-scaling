"""Controller loop for Tunix: poll the orchestrator exporter's /status, decide with BandPolicy, actuate JobSets.

stdlib only. Environment:
    STATUS_URL              e.g. http://tc-orch-metrics.default.svc.cluster.local:8787
    POLL_INTERVAL_S         default 10
    MIN_WORKERS / MAX_WORKERS   policy min/max (max = base workers + templates = free hosts)
    IDLE_HI / IDLE_LO       band on trainer wait share (exposed_generation_time / step_time_sec); default 0.40 / 0.15
    DWELL_STEPS             default 2 (completed steps = policy versions)
    MIN_ACTION_INTERVAL_S   default 120
    TEMPLATES               comma list worker_id=jobset_name=/path/to/rollout-N.yaml (ordered = index)
    BASE_WORKERS            comma list of worker ids the run started with (never removed)
    JOIN_TIMEOUT_S / EVICT_TIMEOUT_S     default 420 / 180
    GUARD_ENABLED / GUARD_WINDOW / GUARD_MIN_GAIN / GUARD_UNBLOCK_AFTER_STEPS   default 1 / 2 / 0.10 / 0
    QUEUE_MIN               pending-queue gate (requires UTIL_HI): a scale-out is actuated only if the orchestrator's
                            pending-queue depth (time-weighted per step, window mean over the same steps as the util
                            gate) is >= QUEUE_MIN requests; 'slots full but queue empty' = episodes stuck on sandboxes,
                            not a reason to add a sampler. Unset = off. Logged as blocked_by_queue.
    UTIL_HI / UTIL_LO       sampler slot-utilization gate (BandPolicy util_hi/util_lo, GPU-M2 deviation D1); unset = off.
                            Slot utilization = sum(inflight) / sum(max_concurrency) over ACTIVE workers from /status,
                            per step (SlotUtilTracker), window mean over UTIL_WINDOW steps (default DWELL_STEPS) of
                            UTIL_STAT ("peak" default | "mean"); UTIL_DEFAULT_CAP = cap for workers without one.
                            Scale-out additionally needs window value >= UTIL_HI; window value < UTIL_LO for dwell
                            steps proposes a scale-in (min and join-in-flight hold still apply). Every record carries
                            a "gate" entry with the band-only verdict (scale_out requested / blocked_by_util / allowed).
    DECISIONS_JSONL         file path (always also printed to stdout)
    DRY_RUN                 "1" -> decide and log, never apply/delete
    NAMESPACE / KUBECTL_CONTEXT / KUBECTL   default "default" / unset / "kubectl"
    HTTP_TIMEOUT_S          default 10

Queue guards of BandPolicy are neutralized for Tunix by MQ_HI_FRAC=0 (default here) with a zero queue gauge:
mq_frac is 0, which satisfies both "queue not full" (< MQ_LO_FRAC) and "queue full" (>= 0), so the band alone
drives both directions (Tunix is on-policy and exports no sampler-side queue depth).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
from typing import Any

from rl_pool_autoscaler.controller.policy import BandPolicy, Observation, PolicyConfig
from rl_pool_autoscaler.controller.tunix_backend import (
    GainGuard,
    JobSetActuator,
    KubectlClient,
    SlotUtilTracker,
    StatusClient,
    Template,
    TunixStatus,
)


def _env(name: str, default: Any, cast=str) -> Any:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return cast(v)


def parse_templates(spec: str) -> list[Template]:
    out: list[Template] = []
    for i, item in enumerate([x for x in spec.split(",") if x.strip()]):
        wid, js, path = item.strip().split("=", 2)
        with open(path) as fh:
            out.append(Template(index=i, worker_id=wid, jobset_name=js, manifest=fh.read()))
    return out


def config_from_env() -> tuple[PolicyConfig, dict[str, Any]]:
    cfg = PolicyConfig(
        min_standalone=_env("MIN_WORKERS", 1, int),
        max_standalone=_env("MAX_WORKERS", 3, int),
        idle_hi=_env("IDLE_HI", 0.40, float),
        idle_lo=_env("IDLE_LO", 0.15, float),
        dwell_steps=_env("DWELL_STEPS", 2, int),
        min_action_interval_s=_env("MIN_ACTION_INTERVAL_S", 120.0, float),
        mq_lo_frac=_env("MQ_LO_FRAC", 0.8, float),
        mq_hi_frac=_env("MQ_HI_FRAC", 0.0, float),
        util_hi=_env("UTIL_HI", None, float),
        util_lo=_env("UTIL_LO", None, float),
    )
    run = {
        "status_url": _env("STATUS_URL", "http://127.0.0.1:8787").rstrip("/"),
        "poll_interval_s": _env("POLL_INTERVAL_S", 10.0, float),
        "jsonl": _env("DECISIONS_JSONL", "", str),
        "dry_run": _env("DRY_RUN", "0", str) in ("1", "true", "True"),
        "timeout_s": _env("HTTP_TIMEOUT_S", 10.0, float),
        "templates": _env("TEMPLATES", "", str),
        "base_workers": [x for x in _env("BASE_WORKERS", "", str).split(",") if x],
        "join_timeout_s": _env("JOIN_TIMEOUT_S", 420.0, float),
        "evict_timeout_s": _env("EVICT_TIMEOUT_S", 180.0, float),
        "guard_enabled": _env("GUARD_ENABLED", "1", str) in ("1", "true", "True"),
        "guard_window": _env("GUARD_WINDOW", 2, int),
        "guard_min_gain": _env("GUARD_MIN_GAIN", 0.10, float),
        "guard_unblock_after_steps": _env("GUARD_UNBLOCK_AFTER_STEPS", 0, int),
        "util_stat": _env("UTIL_STAT", "peak", str),
        "util_window": _env("UTIL_WINDOW", cfg.dwell_steps, int),
        "util_default_cap": _env("UTIL_DEFAULT_CAP", None, int),
        "queue_min": _env("QUEUE_MIN", None, float),
        "namespace": _env("NAMESPACE", "default", str),
        "kubectl_context": _env("KUBECTL_CONTEXT", None, str),
        "kubectl": _env("KUBECTL", "kubectl", str),
    }
    return cfg, run


class DecisionLog:
    def __init__(self, path: str = "") -> None:
        self._fh = open(path, "a", buffering=1) if path else None  # noqa: SIM115

    def write(self, rec: dict[str, Any]) -> None:
        line = json.dumps(rec, default=str)
        print(line, flush=True)
        if self._fh:
            self._fh.write(line + "\n")


class DryKube:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def apply(self, manifest: str) -> str:
        self.calls.append(("apply", manifest[:80]))
        return "dry-run"

    def delete_jobset(self, name: str) -> str:
        self.calls.append(("delete", name))
        return "dry-run"

    def jobset_exists(self, name: str) -> bool:
        return False


def join_in_flight(st: TunixStatus) -> list[str]:
    """Workers registered but not yet ACTIVE (INITIALIZING / PENDING_WEIGHT_SYNC), whoever started them."""
    return sorted(w for w, v in st.workers.items() if v.state in ("initializing", "pending_weight_sync"))


def observation(st: TunixStatus, actuator: JobSetActuator, now: float,
                tracker: SlotUtilTracker | None = None) -> Observation | None:
    if st.last_completed_step is None or st.wait_ratio is None:
        return None
    joining = join_in_flight(st)
    active = len(st.active_ids())
    # slot gate: BandPolicy computes slot_util = rollout_util * max_concurrent_samples / (active * cspr); feeding
    # rollout_util = window value, max_concurrent_samples = active * cap, cspr = cap makes slot_util == window value
    util = cap = None
    gate_extra: dict[str, Any] = {}
    if tracker is not None:
        # window over steps measured entirely at the CURRENT pool size (a step that straddled a join or an eviction
        # is excluded: its peak was taken against the old capacity)
        util = tracker.window_value(int(st.last_completed_step), active)
        cap = tracker.cap_per_worker(st)
        gate_extra = {"slot_gate": tracker.gate_view(int(st.last_completed_step), active)}
    return Observation(
        t=now,
        param_version=int(st.last_completed_step),
        trainer_idle_ratio=float(st.wait_ratio),
        mq_size=0.0,
        max_queue_size=0.0,
        active_standalone=active,
        desired_standalone=None,
        # never issue a change while a membership change is in flight, ours or not (e.g. a JobSet-recreated
        # pod rejoining after a failure): the pool is about to change without us
        scaling_in_progress=actuator.in_progress or bool(joining),
        rollout_util=util,
        max_concurrent_samples=(active * cap) if (util is not None and cap) else None,
        concurrent_samples_per_replica=cap if (util is not None and cap) else None,
        extra={
            "join_in_flight": joining,
            "step_time_sec": st.step.get("step_time_sec"),
            "exposed_generation_time": st.step.get("exposed_generation_time"),
            "live": st.live_ids(),
            "inflight": {w: v.inflight for w, v in st.workers.items()},
            "pending_queue": st.queue.get("pending"),
            "retries_total": st.retries_total,
            "evictions_total": st.evictions_total,
            **gate_extra,
        },
    )


class TunixController:
    def __init__(self, policy: BandPolicy, client: StatusClient, actuator: JobSetActuator, guard: GainGuard | None,
                 log: DecisionLog, dry_run: bool = False, tracker: SlotUtilTracker | None = None,
                 queue_min: float | None = None) -> None:
        self.policy = policy
        self.queue_min = queue_min
        self.client = client
        self.actuator = actuator
        self.guard = guard
        self.log = log
        self.dry_run = dry_run
        self.tracker = tracker
        self.trajectory: list[dict[str, Any]] = []  # (step, active) when a new step completes
        self._last_step: int | None = None
        self._last_active: list[str] | None = None
        # band-only shadow: consecutive new steps with wait share > idle_hi, reset exactly when the policy's own
        # counters are reset; shows in the JSONL what the band alone asked for while the slot gate held it back
        self._band_hi_count = 0
        self._band_last_version: int | None = None

    def _reset_counters(self, now: float) -> None:
        self.policy.hi_count = 0
        self.policy.lo_count = 0
        self.policy.last_action_t = now
        self._band_hi_count = 0

    def _gate_record(self, obs: Observation, d: Any) -> dict[str, Any] | None:
        if self.tracker is None:
            return None
        c = self.policy.cfg
        if self._band_last_version != obs.param_version:
            self._band_last_version = obs.param_version
            self._band_hi_count = self._band_hi_count + 1 if obs.trainer_idle_ratio > c.idle_hi else 0
        util = obs.slot_util()
        band_requests = self._band_hi_count >= c.dwell_steps
        qv = (obs.extra.get("slot_gate") or {}).get("queue_window_value") if isinstance(obs.extra, dict) else None
        if d.action == "scale_out":
            verdict = "blocked_by_queue" if (self.queue_min is not None and (qv is None or qv < self.queue_min)) else "allowed_by_util"
        elif d.action == "scale_in":
            verdict = "scale_in_by_util" if "slot_util" in d.reason else "scale_in_by_band"
        elif band_requests and d.reason.startswith("within band") and (util is None or util < c.util_hi):
            verdict = "blocked_by_util"
        elif band_requests:
            verdict = f"band_requested_scale_out_held:{d.reason}"
        else:
            verdict = "band_not_requesting"
        return {"stat": self.tracker.stat, "slot_util_window": util, "util_hi": c.util_hi, "util_lo": c.util_lo,
                "queue_window": qv, "queue_min": self.queue_min,
                "band_hi_count": self._band_hi_count, "band": "scale_out" if band_requests else "hold",
                "verdict": verdict, "policy_hi_count": self.policy.hi_count, "policy_lo_count": self.policy.lo_count}

    def run_once(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        rec: dict[str, Any] = {"t": now, "kind": "poll"}
        try:
            st = self.client.get()
        except (urllib.error.URLError, OSError, ValueError) as e:
            rec.update(error=f"status: {e}")
            self.log.write(rec)
            return rec
        rec["status"] = st.summary()
        if self.tracker is not None:
            fin = self.tracker.observe(st, now)
            if fin:
                rec["step_util"] = fin if len(fin) > 1 else fin[0]
            rec["slot_sample"] = self.tracker.last_sample
        done = self.actuator.tick(st)
        if done is not None:
            rec["actuation_done"] = done
            if self.guard is not None and done["kind"] == "scale_out":
                self.guard.on_join_done(st, done)
            # steps completed while the change was in flight were measured with the OLD membership:
            # restart dwell and action spacing from the moment the pool actually changed
            self._reset_counters(now)
            rec["policy_reset"] = "counters and spacing restarted at actuation completion"
        if self.guard is not None:
            ev = self.guard.update(st)
            if ev is not None:
                rec["guard_eval"] = ev
            rec["guard_blocked"] = self.guard.blocked
        active_now = st.active_ids()
        if self._last_active is not None and active_now != self._last_active and done is None:
            # the ACTIVE set changed without an actuation of ours completing (eviction after a pod loss, or a
            # JobSet-recreated worker rejoining): steps counted so far were measured with the old pool
            self._reset_counters(now)
            rec["membership_changed_externally"] = {"before": self._last_active, "after": active_now,
                                                    "join_in_flight": join_in_flight(st)}
        self._last_active = active_now
        if st.last_completed_step is not None and st.last_completed_step != self._last_step:
            self._last_step = st.last_completed_step
            self.trajectory.append({"step": st.last_completed_step, "active": len(st.active_ids()),
                                    "live": st.live_ids(), "wait_ratio": st.wait_ratio,
                                    "step_time_sec": st.step.get("step_time_sec"),
                                    "exposed_generation_time": st.step.get("exposed_generation_time"), "t": now})
            rec["new_step"] = True
        obs = observation(st, self.actuator, now, self.tracker)
        if obs is None:
            rec.update(note="no completed step yet")
            self.log.write(rec)
            return rec
        d = self.policy.observe(obs)
        rec["decision"] = d.as_dict()
        gate = self._gate_record(obs, d)
        if gate is not None:
            rec["gate"] = gate
        if d.action in ("scale_out", "scale_in"):
            self._band_hi_count = 0  # the policy reset its own counters on an action
        act: dict[str, Any] | None = None
        if d.action == "scale_out":
            qv = (obs.extra.get("slot_gate") or {}).get("queue_window_value")
            if self.queue_min is not None and (qv is None or qv < self.queue_min):
                # queue term: slots are full but nothing is waiting to be dispatched -> the pool is not falling
                # behind, it is stuck (sandbox tails); do not add a sampler
                act = {"result": "blocked_by_queue", "queue_window": qv, "queue_min": self.queue_min}
            elif self.guard is not None and self.guard.blocked:
                act = {"result": "blocked_by_guard", "guard": self.guard.last_eval}
            elif self.guard is not None and self.guard.pending is not None:
                act = {"result": "awaiting_guard_eval", "pending": self.guard.pending}
            elif not self.actuator.can_scale_out(st):
                act = {"result": "no_free_template_or_host", "failed_templates": sorted(self.actuator.failed_templates)}
            else:
                try:
                    act = self.actuator.scale_out(st)
                    if self.guard is not None:
                        self.guard.on_scale_out_issued(st, act["worker_id"])
                except Exception as e:  # noqa: BLE001
                    act = {"result": "error", "error": str(e)}
        elif d.action == "scale_in":
            if not self.actuator.can_scale_in(st):
                act = {"result": "no_managed_active_worker"}
            else:
                try:
                    act = self.actuator.scale_in(st)
                    if self.guard is not None:
                        self.guard.on_scale_in()
                except Exception as e:  # noqa: BLE001
                    act = {"result": "error", "error": str(e)}
        if act is not None:
            rec["actuation"] = act
        self.log.write(rec)
        return rec


def main() -> int:
    cfg, run = config_from_env()
    policy = BandPolicy(cfg)
    templates = parse_templates(run["templates"])
    kube = DryKube() if run["dry_run"] else KubectlClient(run["namespace"], run["kubectl_context"], run["kubectl"])
    actuator = JobSetActuator(kube, templates, set(run["base_workers"]), run["join_timeout_s"], run["evict_timeout_s"])
    guard = GainGuard(run["guard_window"], run["guard_min_gain"], run["guard_unblock_after_steps"]) if run["guard_enabled"] else None
    tracker = SlotUtilTracker(run["util_stat"], run["util_window"], run["util_default_cap"]) if cfg.util_hi is not None else None
    client = StatusClient(run["status_url"], run["timeout_s"])
    log = DecisionLog(run["jsonl"])
    log.write({"t": time.time(), "kind": "start", "policy": cfg.__dict__, "run": {k: v for k, v in run.items()},
               "templates": [{"index": t.index, "worker_id": t.worker_id, "jobset": t.jobset_name} for t in templates]})
    ctl = TunixController(policy, client, actuator, guard, log, run["dry_run"], tracker, run["queue_min"])
    while True:
        ctl.run_once()
        time.sleep(run["poll_interval_s"])


if __name__ == "__main__":
    sys.exit(main())
