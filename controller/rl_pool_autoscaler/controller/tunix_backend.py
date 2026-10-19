"""Tunix backend for the pool controller: /status client, JobSet actuator, no-gain guard. stdlib only.

Tunix (atwigg/mlperf, PR #2767 membership) has one JobSet per rollout worker and no scale verb, so
    scale-out  = apply one more pre-rendered rollout JobSet (same template as the running ones, unique
                 worker_id / JobSet name, discovery target = the orchestrator JobSet DNS name) and wait until
                 the orchestrator's /status shows that worker_id ACTIVE (registry: INITIALIZING ->
                 PENDING_WEIGHT_SYNC -> ACTIVE at the next committed weight-sync round)
    scale-in   = pick the ACTIVE controller-managed worker with the fewest in-flight requests (tie: newest),
                 delete its JobSet, wait until /status shows it EVICTED (the orchestrator detects the loss on
                 the next dispatch/poll RPC failure and re-queues its in-flight requests)
The base worker(s) that the run was started with are never removed (policy min >= 1).

The exporter JSON (/status) is produced by tunix_exporter.Exporter inside the orchestrator process.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

LIVE_STATES = ("initializing", "pending_weight_sync", "active")


# --------------------------------------------------------------------------------------------- status
@dataclass
class WorkerView:
    worker_id: str
    state: str
    incarnation: int
    inflight: int = 0
    max_concurrency: int | None = None
    state_times: dict[str, float] = field(default_factory=dict)


@dataclass
class TunixStatus:
    t: float
    last_completed_step: int | None
    step_in_progress: int | None
    step: dict[str, Any]
    workers: dict[str, WorkerView]
    counts: dict[str, Any]
    ft: dict[str, Any]
    history: list[dict[str, Any]]
    queue: dict[str, Any] = field(default_factory=dict)
    weight_sync: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "TunixStatus":
        workers = {}
        for wid, w in (d.get("workers") or {}).items():
            workers[wid] = WorkerView(
                worker_id=wid,
                state=str(w.get("state", "unknown")),
                incarnation=int(w.get("incarnation") or 0),
                inflight=int(w.get("inflight") or 0),
                max_concurrency=w.get("max_concurrency"),
                state_times=dict(w.get("state_times") or {}),
            )
        return cls(
            t=float(d.get("t") or time.time()),
            last_completed_step=d.get("last_completed_step"),
            step_in_progress=d.get("step_in_progress"),
            step=dict(d.get("step") or {}),
            workers=workers,
            counts=dict(d.get("counts") or {}),
            ft=dict(d.get("fault_tolerance") or {}),
            history=list(d.get("history") or []),
            queue=dict(d.get("rollout_queue") or {}),
            weight_sync=dict(d.get("weight_sync") or {}),
        )

    @property
    def wait_ratio(self) -> float | None:
        return self.step.get("trainer_wait_ratio")

    @property
    def retries_total(self) -> int:
        return int(self.ft.get("rollout_retries_total") or 0)

    @property
    def evictions_total(self) -> int:
        return int(self.ft.get("rollout_worker_evictions_total") or 0)

    def active_ids(self) -> list[str]:
        return sorted(w for w, v in self.workers.items() if v.state == "active")

    def live_ids(self) -> list[str]:
        return sorted(w for w, v in self.workers.items() if v.state in LIVE_STATES)

    def summary(self) -> dict[str, Any]:
        return {
            "last_completed_step": self.last_completed_step,
            "step_in_progress": self.step_in_progress,
            "step_time_sec": self.step.get("step_time_sec"),
            "exposed_generation_time": self.step.get("exposed_generation_time"),
            "trainer_wait_ratio": self.wait_ratio,
            "policy_training_time": self.step.get("policy_training_time"),
            "weight_sync_time": self.step.get("weight_sync_time"),
            "tokens_per_sec": self.step.get("tokens_per_sec"),
            "workers": {w: f"{v.state}/inc{v.incarnation}/inflight{v.inflight}" for w, v in sorted(self.workers.items())},
            "counts": self.counts,
            "queue": self.queue,
            "ft": self.ft,
        }


class StatusClient:
    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def get(self) -> TunixStatus:
        with urllib.request.urlopen(f"{self.base_url}/status", timeout=self.timeout_s) as r:  # noqa: S310
            return TunixStatus.from_json(json.loads(r.read().decode()))

    def metrics_text(self) -> str:
        with urllib.request.urlopen(f"{self.base_url}/metrics", timeout=self.timeout_s) as r:  # noqa: S310
            return r.read().decode()


# --------------------------------------------------------------------------------------------- kube
class KubeClient(Protocol):
    def apply(self, manifest: str) -> str: ...
    def delete_jobset(self, name: str) -> str: ...
    def jobset_exists(self, name: str) -> bool: ...


class KubectlClient:
    """Shells out to kubectl (in-cluster ServiceAccount, or --context when given)."""

    def __init__(self, namespace: str = "default", context: str | None = None, kubectl: str = "kubectl") -> None:
        self.namespace = namespace
        self.base = [kubectl] + (["--context", context] if context else []) + ["-n", namespace]

    def _run(self, args: list[str], stdin: str | None = None) -> str:
        p = subprocess.run(self.base + args, input=stdin, capture_output=True, text=True, timeout=60, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"kubectl {' '.join(args)} failed rc={p.returncode}: {p.stderr.strip()[:400]}")
        return p.stdout.strip()

    def apply(self, manifest: str) -> str:
        return self._run(["apply", "-f", "-"], stdin=manifest)

    def delete_jobset(self, name: str) -> str:
        return self._run(["delete", "jobset", name, "--wait=false", "--ignore-not-found=true"])

    def jobset_exists(self, name: str) -> bool:
        out = self._run(["get", "jobset", name, "--ignore-not-found=true", "-o", "name"])
        return bool(out)


# --------------------------------------------------------------------------------------------- actuator
@dataclass
class Template:
    index: int
    worker_id: str
    jobset_name: str
    manifest: str


class JobSetActuator:
    """One in-flight membership change at a time; non-blocking (tick() per poll)."""

    def __init__(
        self,
        kube: KubeClient,
        templates: list[Template],
        base_workers: set[str] | None = None,
        join_timeout_s: float = 420.0,
        evict_timeout_s: float = 180.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.kube = kube
        self.templates = sorted(templates, key=lambda t: t.index)
        self.base_workers = set(base_workers or ())
        self.join_timeout_s = join_timeout_s
        self.evict_timeout_s = evict_timeout_s
        self.clock = clock
        self.op: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.failed_templates: set[str] = set()

    # -- views
    @property
    def in_progress(self) -> bool:
        return self.op is not None

    def managed_live(self, st: TunixStatus) -> list[Template]:
        return [t for t in self.templates if st.workers.get(t.worker_id) and st.workers[t.worker_id].state in LIVE_STATES]

    def managed_active(self, st: TunixStatus) -> list[Template]:
        return [t for t in self.templates if st.workers.get(t.worker_id) and st.workers[t.worker_id].state == "active"]

    def next_template(self, st: TunixStatus) -> Template | None:
        live = {t.worker_id for t in self.managed_live(st)}
        for t in self.templates:
            if t.worker_id in live or t.worker_id in self.failed_templates:
                continue
            return t
        return None

    def can_scale_out(self, st: TunixStatus) -> bool:
        return self.op is None and self.next_template(st) is not None

    def can_scale_in(self, st: TunixStatus) -> bool:
        return self.op is None and bool(self.managed_active(st))

    def pick_scale_in(self, st: TunixStatus) -> Template | None:
        cands = self.managed_active(st)
        if not cands:
            return None
        # fewest in-flight first; tie -> newest (highest index)
        return sorted(cands, key=lambda t: (st.workers[t.worker_id].inflight, -t.index))[0]

    # -- actions
    def scale_out(self, st: TunixStatus) -> dict[str, Any]:
        if self.op is not None:
            raise RuntimeError("operation in progress")
        t = self.next_template(st)
        if t is None:
            raise RuntimeError("no free template")
        now = self.clock()
        w = st.workers.get(t.worker_id)
        self.kube.apply(t.manifest)
        self.op = {
            "kind": "scale_out", "worker_id": t.worker_id, "jobset": t.jobset_name, "index": t.index,
            "t_start": now, "step_at_start": st.last_completed_step, "inc_before": (w.incarnation if w else 0),
            "retries_before": st.retries_total, "evictions_before": st.evictions_total,
            "active_before": len(st.active_ids()), "wait_before": st.step.get("exposed_generation_time"),
        }
        return dict(self.op)

    def scale_in(self, st: TunixStatus) -> dict[str, Any]:
        if self.op is not None:
            raise RuntimeError("operation in progress")
        t = self.pick_scale_in(st)
        if t is None:
            raise RuntimeError("no controller-managed ACTIVE worker to remove")
        now = self.clock()
        w = st.workers[t.worker_id]
        self.kube.delete_jobset(t.jobset_name)
        self.op = {
            "kind": "scale_in", "worker_id": t.worker_id, "jobset": t.jobset_name, "index": t.index,
            "t_start": now, "step_at_start": st.last_completed_step, "inc_before": w.incarnation,
            "inflight_at_delete": w.inflight, "retries_before": st.retries_total, "evictions_before": st.evictions_total,
            "active_before": len(st.active_ids()), "wait_before": st.step.get("exposed_generation_time"),
            "candidates": {c.worker_id: st.workers[c.worker_id].inflight for c in self.managed_active(st)},
        }
        return dict(self.op)

    def tick(self, st: TunixStatus) -> dict[str, Any] | None:
        """Advance the in-flight op; returns the completed event (ok or failed) or None."""
        if self.op is None:
            return None
        op = self.op
        now = self.clock()
        w = st.workers.get(op["worker_id"])
        elapsed = now - op["t_start"]
        done: dict[str, Any] | None = None
        if op["kind"] == "scale_out":
            if w is not None and w.state == "active" and w.incarnation > op["inc_before"]:
                done = dict(op, result="ok", t_end=now, join_latency_s=elapsed, incarnation=w.incarnation,
                            state_times=dict(w.state_times), step_at_end=st.last_completed_step,
                            active_after=len(st.active_ids()))
            elif elapsed > self.join_timeout_s:
                try:
                    self.kube.delete_jobset(op["jobset"])
                    cleanup = "jobset deleted"
                except Exception as e:  # noqa: BLE001
                    cleanup = f"delete failed: {e}"
                self.failed_templates.add(op["worker_id"])
                done = dict(op, result="failed", t_end=now, reason=f"not ACTIVE within {self.join_timeout_s:.0f} s "
                            f"(state={w.state if w else 'absent'})", cleanup=cleanup, step_at_end=st.last_completed_step,
                            active_after=len(st.active_ids()))
        else:  # scale_in
            gone = w is None or w.state == "evicted" or w.incarnation != op["inc_before"]
            if gone:
                done = dict(op, result="ok", t_end=now, evict_latency_s=elapsed,
                            requeued=st.retries_total - op["retries_before"],
                            evictions_delta=st.evictions_total - op["evictions_before"],
                            step_at_end=st.last_completed_step, active_after=len(st.active_ids()))
            elif elapsed > self.evict_timeout_s:
                done = dict(op, result="failed", t_end=now, reason=f"not EVICTED within {self.evict_timeout_s:.0f} s "
                            f"(state={w.state})", requeued=st.retries_total - op["retries_before"],
                            step_at_end=st.last_completed_step, active_after=len(st.active_ids()))
        if done is not None:
            self.op = None
            self.events.append(done)
        return done


# --------------------------------------------------------------------------------------------- guard
class GainGuard:
    """Do not keep adding workers when the last join did not lower the trainer wait.

    before = mean exposed_generation_time (s) over the last `window` completed steps when the scale-out was
    issued; after = mean over the first `window` completed steps that ran entirely with the new worker (step
    index > step in progress at promotion + 1). gain = 1 - after / before. If gain < min_gain the guard blocks
    further scale-outs until a scale-in happens (or `unblock_after_steps` more steps pass, if > 0).
    """

    def __init__(self, window: int = 2, min_gain: float = 0.10, unblock_after_steps: int = 0) -> None:
        self.window = window
        self.min_gain = min_gain
        self.unblock_after_steps = unblock_after_steps
        self.blocked = False
        self.pending: dict[str, Any] | None = None
        self.last_eval: dict[str, Any] | None = None
        self.evals: list[dict[str, Any]] = []
        self._blocked_at_step: int | None = None

    @staticmethod
    def _wait_mean(history: list[dict[str, Any]], steps: list[int]) -> float | None:
        vals = [h.get("exposed_generation_time") for h in history if h.get("step") in steps]
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    # step 0 of a run includes compilation (and, on DeepSWE, the sandbox warm-up) in its exposed wait, so it never
    # enters a guard window (seen live on run te: step 0 wait 295.6 s vs 49.8 s at step 1)
    MIN_STEP = 1

    def on_scale_out_issued(self, st: TunixStatus, worker_id: str) -> None:
        steps = [h["step"] for h in st.history if "step" in h and h["step"] >= self.MIN_STEP][-self.window:]
        self.pending = {"worker_id": worker_id, "before_steps": steps, "before": self._wait_mean(st.history, steps),
                        "active_before": len(st.active_ids())}

    def on_join_done(self, st: TunixStatus, event: dict[str, Any]) -> None:
        if self.pending is None or self.pending.get("worker_id") != event.get("worker_id"):
            return
        if event.get("result") != "ok":
            self.pending = None
            return
        # the step in progress at promotion is contaminated; first clean step is step_in_progress + 1
        sip = st.step_in_progress if st.step_in_progress is not None else (st.last_completed_step or -1) + 1
        self.pending["first_clean_step"] = int(sip) + 1

    def on_scale_in(self) -> None:
        self.blocked = False
        self._blocked_at_step = None
        self.pending = None

    def update(self, st: TunixStatus) -> dict[str, Any] | None:
        if self.blocked and self.unblock_after_steps > 0 and st.last_completed_step is not None \
                and self._blocked_at_step is not None and st.last_completed_step - self._blocked_at_step >= self.unblock_after_steps:
            self.blocked = False
            self._blocked_at_step = None
        p = self.pending
        if p is None or "first_clean_step" not in p:
            return None
        steps = [h["step"] for h in st.history if "step" in h and h["step"] >= max(p["first_clean_step"], self.MIN_STEP)][: self.window]
        if len(steps) < self.window:
            return None
        after = self._wait_mean(st.history, steps)
        before = p.get("before")
        gain = (1.0 - after / before) if (before and after is not None and before > 0) else None
        blocked = gain is not None and gain < self.min_gain
        ev = {"kind": "guard_eval", "worker_id": p["worker_id"], "before_steps": p["before_steps"], "before_wait_s": before,
              "after_steps": steps, "after_wait_s": after, "gain": gain, "min_gain": self.min_gain, "blocked": blocked,
              "active_before": p.get("active_before"), "t": st.t}
        self.blocked = blocked
        self._blocked_at_step = st.last_completed_step if blocked else None
        self.last_eval = ev
        self.evals.append(ev)
        self.pending = None
        return ev


# --------------------------------------------------------------------------------------------- slot gate
class SlotUtilTracker:
    """Pool slot utilization per completed step, from the exporter's per-worker in-flight and cap (Tunix side of the
    GPU-M2 slot-utilization gate, M2-REPORT deviation D1).

    At every poll: u = sum(inflight) / sum(max_concurrency) over ACTIVE workers (workers with no advertised cap use
    ``default_cap``; if no cap is known the sample is dropped). Samples taken while ``last_completed_step == s`` are the
    in-flight work of step s + 1 (Tunix dispatches batch s + 1 after the end-of-step-s sync); when ``last_completed_step``
    advances the step is finalized with
        mean  = hold-last time-weighted mean of u over the step   (the GPU side's time-weighted definition)
        peak  = max u over the step                                (= batch / pool cap at dispatch on this on-policy recipe)
        pending_peak, inflight_peak, cap, n_samples, active range
    ``window_value(step)`` = arithmetic mean of ``stat`` (peak or mean) over the last ``window`` finalized steps ending at
    ``step``; the controller feeds it to BandPolicy as the slot utilization (policy.py untouched). Why ``peak`` is the
    default on Tunix: dispatch is batch-synchronous (max_staleness 0), so in-flight falls to the slow tail inside every
    step and the time-weighted mean is bounded by the tail regardless of demand; the peak is demand / slots offered,
    which is what the gate asks (would another host's slots be used?). Both statistics are recorded per step.
    """

    def __init__(self, stat: str = "peak", window: int = 2, default_cap: int | None = None, min_step: int = 0) -> None:
        if stat not in ("peak", "mean"):
            raise ValueError("stat must be 'peak' or 'mean'")
        self.stat = stat
        self.window = max(1, int(window))
        self.default_cap = default_cap
        self.min_step = min_step
        self.per_step: dict[int, dict[str, Any]] = {}
        self._cur_step: int | None = None  # step whose in-flight work is being sampled
        self._samples: list[dict[str, Any]] = []
        self.last_sample: dict[str, Any] | None = None

    @staticmethod
    def _step_in_flight(st: TunixStatus) -> int:
        return (int(st.last_completed_step) + 1) if st.last_completed_step is not None else 0

    def instant(self, st: TunixStatus, now: float) -> dict[str, Any]:
        """One sample: utilization of the ACTIVE pool right now."""
        inflight = 0
        cap = 0
        n_active = 0
        n_nocap = 0
        for v in st.workers.values():
            if v.state != "active":
                continue
            n_active += 1
            inflight += int(v.inflight or 0)
            c = v.max_concurrency if v.max_concurrency is not None else self.default_cap
            if c is None:
                n_nocap += 1
            else:
                cap += int(c)
        u = (inflight / cap) if cap > 0 else None
        return {"t": now, "step": self._step_in_flight(st), "active": n_active, "inflight": inflight, "cap": cap,
                "util": u, "pending": st.queue.get("pending"), "workers_without_cap": n_nocap}

    def observe(self, st: TunixStatus, now: float) -> list[dict[str, Any]]:
        """Record one poll; returns the list of steps finalized by this poll (0 or 1 entries, more only after a gap)."""
        s = self._step_in_flight(st)
        finalized: list[dict[str, Any]] = []
        if self._cur_step is None:
            self._cur_step = s
        while self._cur_step < s:
            fin = self._finalize(self._cur_step, now)
            if fin is not None:
                finalized.append(fin)
            self._cur_step += 1
            self._samples = []
        sample = self.instant(st, now)
        self.last_sample = sample
        if sample["util"] is not None:
            self._samples.append(sample)
        return finalized

    def _finalize(self, step: int, t_end: float) -> dict[str, Any] | None:
        xs = self._samples
        if not xs:
            rec = {"step": step, "n_samples": 0, "mean": None, "peak": None, "pending_peak": None, "pending_mean": None, "inflight_peak": None,
                   "cap": None, "active_min": None, "active_max": None, "t_first": None, "t_last": t_end}
            self.per_step[step] = rec
            return rec
        wsum = 0.0
        usum = 0.0
        psum = 0.0
        for i, x in enumerate(xs):
            t_next = xs[i + 1]["t"] if i + 1 < len(xs) else t_end
            w = max(0.0, float(t_next) - float(x["t"]))
            if w == 0.0 and len(xs) == 1:
                w = 1.0
            wsum += w
            usum += w * float(x["util"])
            psum += w * float(x.get("pending") or 0)
        mean = (usum / wsum) if wsum > 0 else sum(float(x["util"]) for x in xs) / len(xs)
        pending_mean = (psum / wsum) if wsum > 0 else sum(float(x.get("pending") or 0) for x in xs) / len(xs)
        peak_x = max(xs, key=lambda x: x["util"])
        pend = [x["pending"] for x in xs if x.get("pending") is not None]
        rec = {"step": step, "n_samples": len(xs), "mean": mean, "peak": float(peak_x["util"]),
               "inflight_peak": int(peak_x["inflight"]), "cap": int(peak_x["cap"]),
               "pending_peak": (max(pend) if pend else None), "pending_mean": pending_mean,
               "active_min": min(x["active"] for x in xs), "active_max": max(x["active"] for x in xs),
               "t_first": xs[0]["t"], "t_last": t_end}
        self.per_step[step] = rec
        return rec

    def window_steps(self, step: int | None, active: int | None = None) -> list[int]:
        """Steps in the window whose samples were all taken at ONE pool size (and, if ``active`` is given, at the
        current one). A step during which a worker joined or left mixes two pools: its peak was measured against
        the old capacity (seen live on run tg2: step 3 peaked at 64/64 on the single host while the second host
        was promoted mid-step, and the window mean (1.0 + 0.5) / 2 = 0.75 let a third host through)."""
        if step is None:
            return []
        lo = max(self.min_step, step - self.window + 1)
        out = []
        for s in range(lo, step + 1):
            r = self.per_step.get(s)
            if r is None or r.get(self.stat) is None:
                continue
            if r.get("active_min") != r.get("active_max"):
                continue
            if active is not None and r.get("active_max") != active:
                continue
            out.append(s)
        return out

    def window_value(self, step: int | None, active: int | None = None) -> float | None:
        steps = self.window_steps(step, active)
        if not steps:
            return None
        return sum(float(self.per_step[s][self.stat]) for s in steps) / len(steps)

    def queue_window_value(self, step: int | None, active: int | None = None) -> float | None:
        """Mean of the per-step time-weighted pending-queue depth over the same window as the utilization value.
        Queue term (coordinator design change after tg11 step 11): 'slots full AND queue deep' = falling behind,
        'slots full AND queue empty' = stuck on sandboxes (episodes hold their slots while they wait); only the
        first is a reason to add a sampler."""
        steps = self.window_steps(step, active)
        vals = [self.per_step[s].get("pending_mean") for s in steps]
        vals = [float(v) for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    def cap_per_worker(self, st: TunixStatus) -> float | None:
        caps = [float(v.max_concurrency if v.max_concurrency is not None else (self.default_cap or 0))
                for v in st.workers.values() if v.state == "active"]
        caps = [c for c in caps if c > 0]
        return (sum(caps) / len(caps)) if caps else None

    def gate_view(self, step: int | None, active: int | None = None) -> dict[str, Any]:
        steps = self.window_steps(step, active)
        return {"stat": self.stat, "window": self.window, "window_steps": steps, "window_value": self.window_value(step, active),
                "excluded_mixed_pool_steps": [s for s in range(max(self.min_step, (step or 0) - self.window + 1), (step or 0) + 1)
                                              if s in self.per_step and s not in steps and self.per_step[s].get(self.stat) is not None],
                "queue_window_value": self.queue_window_value(step, active),
                "per_step": {s: {k: self.per_step[s].get(k) for k in ("mean", "peak", "inflight_peak", "cap", "pending_peak", "pending_mean", "n_samples")}
                             for s in steps}}
