"""In-process metrics exporter for the Tunix orchestrator (observability only; no algorithm change).

Started by the (observability-patched) example main() right after the ClusterOrchestrator exists:

    exporter = tunix_exporter.Exporter(cluster, port=8787)
    jax.monitoring.register_scalar_listener(exporter.on_scalar)
    cluster.registry.add_state_listener(exporter.on_membership)
    ... on_step_begin -> exporter.on_step_begin(step); on_step_end -> exporter.on_step_end(step)
    exporter.serve()

Routes (stdlib ThreadingHTTPServer, daemon thread):
    GET /healthz   200 "ok"
    GET /metrics   Prometheus text (tunix_* series, see metrics_text)
    GET /status    JSON: last completed step metrics, per-worker membership (state/incarnation/in-flight/cap),
                   counts, engine fault_tolerance_metrics, weight-sync flags, recent membership events, history.

Sources (all read-only):
    * per-step scalars from jax.monitoring (the same values MetricsLogger writes to TensorBoard/W&B):
      orchestrator/<mode>/{step_time_sec,exposed_generation_time,policy_training_time,weight_sync_time,
      metrics_fetch_time,num_rollouts,policy_version}, rollout/<mode>/{global_valid_toks,completions/mean_length,...},
      rewards/<mode>/mean. A step's values are published when on_step_end(step) fires (after the program logged them).
    * membership from cluster.registry (worker_ids(include_evicted=True), state, incarnation, info().resources)
    * per-worker in-flight from engine._rollout_session._dispatched_tasks (private; wrapped in try/except),
      pending queue depth from engine._rollout_session._pending_queue
    * engine.fault_tolerance_metrics, weight_sync_coordinator.in_flight / has_pending_destinations
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

STEP_SUFFIXES = {
    "orchestrator/step_time_sec": "step_time_sec",
    "orchestrator/exposed_generation_time": "exposed_generation_time",
    "orchestrator/policy_training_time": "policy_training_time",
    "orchestrator/weight_sync_time": "weight_sync_time",
    "orchestrator/metrics_fetch_time": "metrics_fetch_time",
    "orchestrator/num_rollouts": "num_rollouts",
    "orchestrator/num_microbatches": "num_microbatches",
    "orchestrator/policy_version": "policy_version",
    "rollout/global_valid_toks": "valid_tokens",
    "rollout/global_valid_seqs": "valid_seqs",
    "rollout/completions/mean_length": "completion_mean_length",
    "rollout/completions/clip_ratio": "completion_clip_ratio",
    "rollout/total_tokens_mean": "total_tokens_mean",
    "rollout/filtered_trajectories_count": "filtered_trajectories",
    "rewards/mean": "reward_mean",
    "perf/global_step_time": "global_step_time",
}


def _match_suffix(name: str) -> str | None:
    # names arrive as "<prefix>/<mode>/<rest>" e.g. orchestrator/train/step_time_sec
    parts = name.split("/")
    if len(parts) >= 3:
        stripped = parts[0] + "/" + "/".join(parts[2:])
        if stripped in STEP_SUFFIXES:
            return STEP_SUFFIXES[stripped]
    return STEP_SUFFIXES.get(name)


def _fmt(v: Any) -> str:
    if v is None:
        return "NaN"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    try:
        return repr(float(v))
    except (TypeError, ValueError):
        return "NaN"


def _labels(labels: dict[str, Any] | None) -> str:
    if not labels:
        return ""
    inner = ",".join(
        f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for k, v in labels.items()
    )
    return "{" + inner + "}"


class Exporter:
    """Thread-safe snapshot of orchestrator state + HTTP server."""

    def __init__(self, cluster: Any, port: int = 8787, run_name: str = "", history: int = 1000) -> None:
        self.cluster = cluster
        self.port = int(port)
        self.run_name = run_name
        self._lock = threading.Lock()
        self._pending: dict[int, dict[str, float]] = collections.defaultdict(dict)
        self._last: dict[str, float] = {}
        self._completed_step: int | None = None
        self._step_in_progress: int | None = None
        self._t_step_begin: float | None = None
        self._t_step_end: float | None = None
        self._history: collections.deque = collections.deque(maxlen=history)
        self._events: collections.deque = collections.deque(maxlen=history)
        self._state_t: dict[str, dict[str, float]] = collections.defaultdict(dict)  # worker -> state -> t
        self._t0 = time.time()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._log = logging.getLogger("tunix_exporter")

    # ---- hooks -------------------------------------------------------------------------------
    def on_scalar(self, name: str, value: Any, **kw: Any) -> None:
        key = _match_suffix(str(name))
        if key is None:
            return
        step = kw.get("step")
        try:
            step_i = int(step)
            val = float(value)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._pending[step_i][key] = val

    def on_step_begin(self, step: int) -> None:
        with self._lock:
            self._step_in_progress = int(step)
            self._t_step_begin = time.time()

    def on_step_end(self, step: int) -> None:
        step = int(step)
        now = time.time()
        with self._lock:
            d = dict(self._pending.pop(step, {}))
            d["step"] = step
            d["t_end"] = now
            d["t_begin"] = self._t_step_begin
            wall = d.get("step_time_sec")
            wait = d.get("exposed_generation_time")
            d["trainer_wait_ratio"] = (wait / wall) if (wall and wall > 0 and wait is not None) else None
            toks = d.get("valid_tokens")
            d["tokens_per_sec"] = (toks / wall) if (wall and wall > 0 and toks is not None) else None
            try:
                d["active_rollout_workers"] = self._engine_pool_size_locked()
            except Exception:  # noqa: BLE001
                d["active_rollout_workers"] = None
            self._last = d
            self._completed_step = step
            self._t_step_end = now
            self._history.append(d)
            # drop stale partials from older steps
            for k in [k for k in self._pending if k < step]:
                self._pending.pop(k, None)

    def on_membership(self, worker_id: str, state: Any) -> None:
        now = time.time()
        sval = getattr(state, "value", state)
        inc = None
        try:
            inc = self.cluster.registry.incarnation(worker_id)
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            self._events.append({"t": now, "worker_id": worker_id, "state": str(sval), "incarnation": inc,
                                 "step_in_progress": self._step_in_progress, "last_completed_step": self._completed_step})
            if str(sval) in ("initializing",):
                self._state_t[worker_id] = {}
            self._state_t[worker_id][str(sval)] = now

    # ---- readers -----------------------------------------------------------------------------
    def _engine(self) -> Any:
        return getattr(self.cluster, "engine", None)

    def _engine_pool_size_locked(self) -> int | None:
        eng = self._engine()
        if eng is None:
            return None
        try:
            return len(eng._rollout_workers)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            return None

    def _inflight_by_worker(self) -> tuple[dict[str, int], int | None, int | None]:
        eng = self._engine()
        if eng is None:
            return {}, None, None
        per: dict[str, int] = {}
        pending = None
        total = None
        try:
            sess = eng._rollout_session  # noqa: SLF001
            for handle, tasks in list(getattr(sess, "_dispatched_tasks", {}).items()):
                wid = getattr(handle, "worker_id", None) or str(handle)
                per[str(wid)] = len(tasks)
            pq = getattr(sess, "_pending_queue", None)
            pending = len(pq) if pq is not None else None
            total = getattr(sess, "in_flight_count", None)
            if callable(total):
                total = total()
        except Exception:  # noqa: BLE001
            pass
        return per, pending, total

    def workers(self) -> dict[str, dict[str, Any]]:
        reg = self.cluster.registry
        per, _, _ = self._inflight_by_worker()
        out: dict[str, dict[str, Any]] = {}
        try:
            ids = reg.worker_ids(include_evicted=True)
        except Exception:  # noqa: BLE001
            return out
        for wid in ids:
            try:
                info = reg.info(wid)
                roles = sorted(getattr(info, "roles", []) or [])
                if "rollout" not in roles:
                    continue
                res = dict(getattr(info, "resources", {}) or {})
                st = reg.state(wid)
                out[wid] = {
                    "state": str(getattr(st, "value", st)),
                    "incarnation": reg.incarnation(wid),
                    "inflight": per.get(wid, 0),
                    "max_concurrency": res.get("max_concurrency"),
                    "address": res.get("address"),
                    "state_times": dict(self._state_t.get(wid, {})),
                }
            except Exception as e:  # noqa: BLE001
                out[wid] = {"state": "unknown", "error": repr(e)}
        return out

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            last = dict(self._last)
            completed = self._completed_step
            in_prog = self._step_in_progress
            t_end = self._t_step_end
            events = list(self._events)[-200:]
            history = list(self._history)[-200:]
        workers = self.workers()
        counts = collections.Counter(w.get("state", "unknown") for w in workers.values())
        per, pending, total = self._inflight_by_worker()
        eng = self._engine()
        ft: dict[str, Any] = {}
        ws: dict[str, Any] = {}
        pool_size = None
        policy_version = None
        if eng is not None:
            try:
                ft = dict(eng.fault_tolerance_metrics)
            except Exception as e:  # noqa: BLE001
                ft = {"error": repr(e)}
            try:
                pool_size = len(eng._rollout_workers)  # noqa: SLF001
            except Exception:  # noqa: BLE001
                pool_size = None
            try:
                policy_version = int(eng.policy_version)
            except Exception:  # noqa: BLE001
                policy_version = None
            try:
                coord = eng.weight_sync_coordinator
                if coord is not None:
                    ws = {"in_flight": bool(coord.in_flight), "pending_destinations": bool(coord.has_pending_destinations())}
            except Exception as e:  # noqa: BLE001
                ws = {"error": repr(e)}
        return {
            "t": now,
            "run_name": self.run_name,
            "uptime_s": now - self._t0,
            "last_completed_step": completed,
            "step_in_progress": in_prog,
            "seconds_since_step_end": (now - t_end) if t_end else None,
            "policy_version": policy_version,
            "step": last,
            "workers": workers,
            "counts": {"active": counts.get("active", 0), "pending_weight_sync": counts.get("pending_weight_sync", 0),
                       "initializing": counts.get("initializing", 0), "evicted": counts.get("evicted", 0),
                       "engine_pool_size": pool_size},
            "rollout_queue": {"pending": pending, "inflight_total": total, "inflight_by_worker": per},
            "fault_tolerance": ft,
            "weight_sync": ws,
            "events": events,
            "history": history,
        }

    def metrics_text(self) -> str:
        s = self.snapshot()
        lines: list[str] = []

        def g(name: str, value: Any, labels: dict[str, Any] | None = None, help_: str = "") -> None:
            if help_:
                lines.append(f"# HELP {name} {help_}")
                lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{_labels(labels)} {_fmt(value)}")

        g("tunix_exporter_up", 1, help_="exporter alive")
        g("tunix_exporter_uptime_seconds", s["uptime_s"])
        g("tunix_last_completed_step", s["last_completed_step"], help_="last step whose metrics were logged")
        g("tunix_step_in_progress", s["step_in_progress"])
        g("tunix_seconds_since_step_end", s["seconds_since_step_end"])
        g("tunix_policy_version", s["policy_version"])
        st = s["step"]
        g("tunix_step_time_sec", st.get("step_time_sec"), help_="orchestrator/step_time_sec of the last completed step")
        g("tunix_exposed_generation_time_sec", st.get("exposed_generation_time"), help_="trainer wait on the scored queue (orchestrator/exposed_generation_time)")
        g("tunix_policy_training_time_sec", st.get("policy_training_time"))
        g("tunix_weight_sync_time_sec", st.get("weight_sync_time"))
        g("tunix_metrics_fetch_time_sec", st.get("metrics_fetch_time"))
        g("tunix_trainer_wait_ratio", st.get("trainer_wait_ratio"), help_="exposed_generation_time / step_time_sec")
        g("tunix_num_rollouts", st.get("num_rollouts"))
        g("tunix_valid_tokens", st.get("valid_tokens"), help_="rollout/global_valid_toks of the last completed step")
        g("tunix_tokens_per_sec", st.get("tokens_per_sec"), help_="valid_tokens / step_time_sec")
        g("tunix_completion_mean_length", st.get("completion_mean_length"))
        g("tunix_completion_clip_ratio", st.get("completion_clip_ratio"))
        g("tunix_reward_mean", st.get("reward_mean"))
        g("tunix_step_active_rollout_workers", st.get("active_rollout_workers"), help_="engine pool size at the end of the last completed step")
        c = s["counts"]
        first = True
        for state in ("initializing", "pending_weight_sync", "active", "evicted"):
            g("tunix_rollout_workers", c.get(state, 0), {"state": state}, "rollout workers by membership state" if first else "")
            first = False
        g("tunix_rollout_pool_size", c.get("engine_pool_size"), help_="engine rollout pool size (dispatchable workers)")
        first = True
        for wid, w in sorted(s["workers"].items()):
            g("tunix_rollout_worker_info", 1, {"worker_id": wid, "state": w.get("state"), "incarnation": w.get("incarnation"), "address": w.get("address")},
              "per-worker membership state" if first else "")
            first = False
        first = True
        for wid, w in sorted(s["workers"].items()):
            g("tunix_rollout_worker_inflight", w.get("inflight"), {"worker_id": wid}, "dispatched, not yet returned requests per worker" if first else "")
            first = False
        first = True
        for wid, w in sorted(s["workers"].items()):
            g("tunix_rollout_worker_max_concurrency", w.get("max_concurrency"), {"worker_id": wid}, "advertised per-worker in-flight cap" if first else "")
            first = False
        first = True
        for wid, w in sorted(s["workers"].items()):
            g("tunix_rollout_worker_incarnation", w.get("incarnation"), {"worker_id": wid}, "registry incarnation" if first else "")
            first = False
        q = s["rollout_queue"]
        g("tunix_rollout_pending_queue_depth", q.get("pending"), help_="requests queued, not dispatched")
        g("tunix_rollout_inflight_total", q.get("inflight_total"))
        ws = s["weight_sync"]
        g("tunix_weight_sync_in_flight", ws.get("in_flight"))
        g("tunix_weight_sync_pending_destinations", ws.get("pending_destinations"))
        for k, v in sorted(s["fault_tolerance"].items()):
            if isinstance(v, (int, float)):
                g(f"tunix_ft_{k}", v, help_="engine.fault_tolerance_metrics")
        return "\n".join(lines) + "\n"

    # ---- server ------------------------------------------------------------------------------
    def serve(self) -> None:
        exporter = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "tunix-exporter/0.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                try:
                    if path == "/healthz":
                        self._send(200, b"ok\n", "text/plain")
                    elif path == "/metrics":
                        self._send(200, exporter.metrics_text().encode(), "text/plain; version=0.0.4")
                    elif path == "/status":
                        self._send(200, json.dumps(exporter.snapshot(), default=str).encode(), "application/json")
                    else:
                        self._send(404, b'{"error":"not found"}', "application/json")
                except Exception as e:  # noqa: BLE001
                    self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="tunix-exporter", daemon=True)
        self._thread.start()
        self._log.info("EXPORTER listening on :%d (/metrics /status /healthz)", self.port)
        logging.info("EXPORTER listening on :%d (/metrics /status /healthz)", self.port)
