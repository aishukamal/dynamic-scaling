"""Controller loop: poll /metrics, decide, POST /scale, log JSONL. stdlib only.

Environment (all optional except CONTROL_URL):
    CONTROL_URL                 e.g. http://raycluster-head-svc.default.svc:8787
    POLL_INTERVAL_S             default 15
    MIN_STANDALONE / MAX_STANDALONE     default 1 / 8 (should match rl_pool.{min,max}_standalone)
    IDLE_HI / IDLE_LO           default 0.5 / 0.15
    DWELL_STEPS                 default 2 (in param versions)
    MIN_ACTION_INTERVAL_S       default 600
    MQ_LO_FRAC / MQ_HI_FRAC     default 0.8 / 0.9
    UTIL_HI / UTIL_LO           sampler slot-utilization gate (both or neither; default off)
    FLOOR_MIN_GAIN              floor guard: min fractional wait reduction a join must show (default off)
    FLOOR_SKIP_VERSIONS         floor guard ignores waits of param versions <= this (start-up transient; default 0)
    DECISIONS_JSONL             file path for the decision log (also always printed to stdout)
    DRY_RUN                     "1" -> never POST, only log
    HTTP_TIMEOUT_S              default 10
    K8S_ACTUATOR                "1" -> per-replica-pod layout: scale the RayCluster worker group too
    RAYCLUSTER_NAME / RAYCLUSTER_NS / SAMPLER_GROUP   default swe-poc / default / sampler-1g
    NODE_WAIT_TIMEOUT_S         default 1200 (pod schedule + Ray join)
    RELEASE_ON_SCALE_IN         "1" -> POST /scale with release=true (destroy instead of park)

Each poll writes one JSONL record with the full inputs so a regret-vs-oracle analysis can replay
the controller offline.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from rl_pool_autoscaler import prom
from rl_pool_autoscaler.controller.policy import BandPolicy, Observation, PolicyConfig


def _env(name: str, default: Any, cast=str) -> Any:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return cast(v)


def config_from_env() -> tuple[PolicyConfig, dict[str, Any]]:
    cfg = PolicyConfig(
        min_standalone=_env("MIN_STANDALONE", 1, int),
        max_standalone=_env("MAX_STANDALONE", 8, int),
        idle_hi=_env("IDLE_HI", 0.5, float),
        idle_lo=_env("IDLE_LO", 0.15, float),
        dwell_steps=_env("DWELL_STEPS", 2, int),
        min_action_interval_s=_env("MIN_ACTION_INTERVAL_S", 600.0, float),
        mq_lo_frac=_env("MQ_LO_FRAC", 0.8, float),
        mq_hi_frac=_env("MQ_HI_FRAC", 0.9, float),
        util_hi=_env("UTIL_HI", None, float),
        util_lo=_env("UTIL_LO", None, float),
        floor_min_gain=_env("FLOOR_MIN_GAIN", None, float),
        floor_skip_versions=_env("FLOOR_SKIP_VERSIONS", 0, int),
    )
    run = {
        "control_url": _env("CONTROL_URL", "http://127.0.0.1:8787").rstrip("/"),
        "poll_interval_s": _env("POLL_INTERVAL_S", 15.0, float),
        "jsonl": _env("DECISIONS_JSONL", "", str),
        "dry_run": _env("DRY_RUN", "0", str) in ("1", "true", "True"),
        "timeout_s": _env("HTTP_TIMEOUT_S", 10.0, float),
        "k8s_actuator": _env("K8S_ACTUATOR", "0", str) in ("1", "true", "True"),
        "raycluster_name": _env("RAYCLUSTER_NAME", "swe-poc", str),
        "raycluster_ns": _env("RAYCLUSTER_NS", "default", str),
        "sampler_group": _env("SAMPLER_GROUP", "sampler-1g", str),
        "node_wait_timeout_s": _env("NODE_WAIT_TIMEOUT_S", 1200.0, float),
        "release_on_scale_in": _env("RELEASE_ON_SCALE_IN", "0", str) in ("1", "true", "True"),
    }
    return cfg, run


class ControlClient:
    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s

    def metrics(self) -> list[prom.Sample]:
        with urllib.request.urlopen(f"{self.base_url}/metrics", timeout=self.timeout_s) as r:  # noqa: S310
            return prom.parse(r.read().decode())

    def status(self) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.base_url}/status", timeout=self.timeout_s) as r:  # noqa: S310
            return json.loads(r.read().decode() or "{}")

    def scale(self, standalone: int, release: bool | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"standalone": int(standalone)}
        if release is not None:
            body["release"] = bool(release)
        data = json.dumps(body).encode()
        req = urllib.request.Request(f"{self.base_url}/scale", data=data, method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:  # noqa: S310
            return json.loads(r.read().decode() or "{}")


def observation_from_samples(samples: list[prom.Sample], now: float) -> Observation | None:
    version = prom.scalar(samples, "rl_pool_current_param_version")
    if version is None:
        return None
    # Key "new sample" on the last aggregated training log when the plugin exports it (the param version is
    # bumped a few ms before that log lands, so a poll in between would pair a new version with stale step/wait).
    tls = prom.scalar(samples, "rl_pool_trainer_train_log_step")
    if tls is not None and tls == tls:
        version = tls
    elif _num(prom.scalar(samples, "rl_pool_trainer_idle_ratio")) is None and _num(prom.scalar(samples, "rl_pool_trainer_step_s")) is None:
        # no aggregated training log yet (first version in flight): not a sample (seen in m3-ctl, 03:33Z: the
        # bumped current_param_version was consumed as "version 1" with NaN step/wait and util 0.0)
        return None
    desired = prom.scalar(samples, "rl_pool_desired_standalone_replicas")
    return Observation(
        t=now,
        param_version=int(version),
        trainer_idle_ratio=float(prom.scalar(samples, "rl_pool_trainer_idle_ratio", 0.0)),
        mq_size=float(prom.scalar(samples, "rl_pool_mq_size", 0.0)),
        max_queue_size=float(prom.scalar(samples, "rl_pool_max_queue_size", 0.0)),
        active_standalone=int(prom.scalar(samples, "rl_pool_active_standalone_replicas", 0)),
        desired_standalone=None if desired is None or desired != desired else int(desired),
        scaling_in_progress=bool(prom.scalar(samples, "rl_pool_scaling_in_progress", 0)),
        rollout_util=_num(prom.scalar(samples, "rl_pool_rollout_resource_utilization")),
        max_concurrent_samples=_num(prom.scalar(samples, "rl_pool_max_concurrent_samples")),
        concurrent_samples_per_replica=_num(prom.scalar(samples, "rl_pool_concurrent_samples_per_replica")),
        extra={
            "current_param_version": prom.scalar(samples, "rl_pool_current_param_version"),
            "sleeping": prom.scalar(samples, "rl_pool_sleeping_standalone_replicas"),
            "trainer_wait_s": prom.scalar(samples, "rl_pool_trainer_wait_s"),
            "trainer_step_s": prom.scalar(samples, "rl_pool_trainer_step_s"),
            "last_sync_wall_s": prom.scalar(samples, "rl_pool_last_sync_wall_s"),
            "rollouter_idle_ratio": prom.scalar(samples, "rl_pool_rollouter_idle_ratio"),
            "trainer_waiting_now": prom.scalar(samples, "rl_pool_trainer_waiting_now"),
            "inflight": prom.by_label(samples, "rl_pool_replica_inflight_requests", "replica"),
            "lb_total_inflight": prom.scalar(samples, "rl_pool_lb_total_inflight"),
            "active_tasks": prom.scalar(samples, "rl_pool_active_tasks"),
            "response_length_mean": prom.scalar(samples, "rl_pool_response_length_mean"),
            "num_turns_mean": prom.scalar(samples, "rl_pool_num_turns_mean"),
            "score_mean": prom.scalar(samples, "rl_pool_score_mean"),
            "dropped_stale": prom.scalar(samples, "rl_pool_dropped_stale"),
        },
    )


def _num(v):
    """None for missing/NaN samples."""
    if v is None or v != v:
        return None
    return float(v)


class DecisionLog:
    def __init__(self, path: str = "") -> None:
        self.path = path
        self._fh = open(path, "a", buffering=1) if path else None  # noqa: SIM115

    def write(self, rec: dict[str, Any]) -> None:
        line = json.dumps(rec, default=str)
        print(line, flush=True)
        if self._fh:
            self._fh.write(line + "\n")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def run_once(
    policy: BandPolicy,
    client: ControlClient,
    log: DecisionLog,
    dry_run: bool,
    now: float | None = None,
    actuator=None,
    release: bool | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    rec: dict[str, Any] = {"t": now, "kind": "poll"}
    try:
        samples = client.metrics()
    except (urllib.error.URLError, OSError, ValueError) as e:
        rec.update(error=f"metrics: {e}")
        log.write(rec)
        return rec
    obs = observation_from_samples(samples, now)
    if obs is None:
        rec.update(error="metrics missing rl_pool_current_param_version (trainer has not logged yet)")
        log.write(rec)
        return rec

    # Two-phase moves of the pod actuator take precedence over new decisions.
    if actuator is not None and actuator.pending is not None:
        try:
            status = client.status()
        except (urllib.error.URLError, OSError, ValueError) as e:
            rec.update(error=f"status: {e}", actuator=dict(actuator.pending))
            log.write(rec)
            return rec
        pend = actuator.pending
        if pend["kind"] == "out":
            done = actuator.poll_scale_out(status, now)
            if done is not None and done.get("ready") and "error" not in done:
                try:
                    posted = client.scale(done["desired"], release)
                except (urllib.error.URLError, OSError, ValueError) as e:
                    posted = {"error": str(e)}
                actuator.finish({"posted": posted, "t_post_scale": time.time()})
                rec.update(kind="actuator", actuator=actuator.history[-1])
            else:
                rec.update(kind="actuator_wait", actuator=dict(pend) if done is None else done)
        else:
            done = actuator.poll_scale_in(status, now)
            rec.update(kind="actuator" if done is not None else "actuator_wait", actuator=done if done is not None else dict(pend))
        # keep the policy's version bookkeeping current without acting
        policy.observe(obs)
        log.write(rec)
        return rec

    decision = policy.observe(obs)
    rec.update(decision=decision.as_dict())
    if decision.action in ("scale_out", "scale_in"):
        if dry_run:
            rec["posted"] = {"dry_run": True, "standalone": decision.desired}
        elif actuator is not None:
            try:
                if decision.action == "scale_out":
                    status = client.status()
                    a = actuator.begin_scale_out(decision.desired, status, now)
                    if a.get("ready"):
                        rec["posted"] = client.scale(decision.desired, release)
                        actuator.finish({"posted": rec["posted"], "t_post_scale": time.time()})
                        a = actuator.history[-1]
                    rec["actuator"] = dict(a)
                else:
                    a = actuator.begin_scale_in(decision.desired, now)
                    rec["posted"] = client.scale(decision.desired, release)
                    a["posted"] = rec["posted"]
                    rec["actuator"] = dict(a)
            except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
                rec["posted"] = {"error": str(e)}
                actuator.pending = None
        else:
            try:
                rec["posted"] = client.scale(decision.desired, release)
            except (urllib.error.URLError, OSError, ValueError) as e:
                rec["posted"] = {"error": str(e)}
    log.write(rec)
    return rec


def main() -> int:
    cfg, run = config_from_env()
    policy = BandPolicy(cfg)
    client = ControlClient(run["control_url"], run["timeout_s"])
    log = DecisionLog(run["jsonl"])
    actuator = None
    if run["k8s_actuator"]:
        from rl_pool_autoscaler.controller.k8s_actuator import K8sClient, PodActuator, RayClusterGroup

        group = RayClusterGroup(K8sClient(), run["raycluster_ns"], run["raycluster_name"], run["sampler_group"])
        actuator = PodActuator(group, node_wait_timeout_s=run["node_wait_timeout_s"], release=run["release_on_scale_in"])
    release = True if run["release_on_scale_in"] else None
    log.write({"t": time.time(), "kind": "start", "policy": cfg.__dict__, "run": run})
    while True:
        try:
            run_once(policy, client, log, run["dry_run"], actuator=actuator, release=release)
        except Exception as e:  # noqa: BLE001 - keep the loop alive, record the failure
            log.write({"t": time.time(), "kind": "loop_error", "error": f"{type(e).__name__}: {e}"})
        time.sleep(run["poll_interval_s"])


if __name__ == "__main__":
    sys.exit(main())
