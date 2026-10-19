"""RLPoolControl: the named Ray actor ("rl_pool_control") with the HTTP API and the reconciler.

Threads inside the actor process:
- HTTP server (stdlib) serving GET /metrics, GET /status, POST /scale, GET /healthz.
- Reconciler: runs ``ScaleOrchestrator.step()`` until converged whenever desired != active.
Ray methods (callable via ``ray.get_actor("rl_pool_control")``): ``push_trainer_metrics``,
``scale``, ``status``, ``metrics_text``, ``ready``.

Metric sourcing:
- Trainer numbers are PUSHED by ``ElasticTrainer`` whenever it logs (and around queue waits), so
  /metrics never blocks on the trainer actor (whose event loop is blocked during update_actor).
- Rollouter numbers are PULLED (``get_pool_status``) with a short timeout; the rollouter is an
  async actor with max_concurrency=100 and never blocks for long. Last good value is cached.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from rl_pool_autoscaler import prom
from rl_pool_autoscaler.http_api import ControlHTTPServer
from rl_pool_autoscaler.orchestrator import PoolLimits, ScaleOrchestrator


class RayPoolBackend:
    """PoolBackend implementation over the two Ray actors. Every call is a blocking ray.get."""

    def __init__(self, trainer, rollouter, timeout_s: float = 1800.0) -> None:
        import ray

        self._ray = ray
        self.trainer = trainer
        self.rollouter = rollouter
        self.timeout_s = timeout_s

    def _get(self, ref):
        return self._ray.get(ref, timeout=self.timeout_s)

    def list_active(self) -> list[str]:
        return list(self._get(self.rollouter.get_active_standalone_ids.remote()))

    def list_parked(self) -> list[str]:
        return list(self._get(self.rollouter.get_parked_standalone_ids.remote()))

    def create_standalone(self, n: int) -> list[str]:
        return list(self._get(self.rollouter.add_standalone.remote(n)))

    def wake(self, rids: list[str]) -> None:
        self._get(self.rollouter.wake_standalone.remote(rids))

    def resume_generation(self, rids: list[str]) -> None:
        self._get(self.rollouter.resume_standalone.remote(rids))

    def attach(self, rids: list[str]) -> None:
        self._get(self.trainer.attach_replicas.remote(rids))

    def sync_joiners(self, rids: list[str]) -> dict:
        return dict(self._get(self.trainer.sync_joiners_now.remote(rids)) or {})

    def _membership(self) -> dict:
        return dict(self._get(self.trainer.get_membership.remote()) or {})

    def replica_versions(self) -> dict[str, int]:
        return dict(self._membership().get("versions") or {})

    def last_synced_version(self) -> int | None:
        return self._membership().get("last_synced_version")

    def lb_add(self, rids: list[str]) -> None:
        self._get(self.rollouter.lb_add.remote(rids))

    def lb_remove(self, rids: list[str]) -> None:
        self._get(self.rollouter.lb_remove.remote(rids))

    def abort(self, rids: list[str]) -> None:
        self._get(self.rollouter.abort_standalone.remote(rids))

    def detach(self, rids: list[str]) -> None:
        self._get(self.trainer.detach_replicas.remote(rids))

    def sleep(self, rids: list[str]) -> None:
        self._get(self.rollouter.sleep_standalone.remote(rids))

    def destroy(self, rids: list[str]) -> None:
        self._get(self.rollouter.destroy_standalone.remote(rids))

    def evict(self, rids: list[str]) -> None:
        self._get(self.rollouter.evict_standalone.remote(rids))

    def check_health(self, timeout_s: float = 10.0) -> dict:
        return dict(self._ray.get(self.rollouter.check_health.remote(timeout_s), timeout=timeout_s * 4 + 10) or {})


class ControlCore:
    """Everything except Ray actor plumbing, so it can be unit tested with a fake backend.

    ``pool_status_fn`` returns the rollouter's ``get_pool_status()`` dict (or raises).
    """

    def __init__(
        self,
        orchestrator: ScaleOrchestrator,
        pool_status_fn,
        nodes_fn=None,
        status_cache_s: float = 2.0,
        reconcile_idle_s: float = 5.0,
        error_backoff_s: float = 30.0,
        clock=time.time,
    ) -> None:
        self.orch = orchestrator
        self._pool_status_fn = pool_status_fn
        self._nodes_fn = nodes_fn
        self._status_cache_s = status_cache_s
        self._reconcile_idle_s = reconcile_idle_s
        self._error_backoff_s = error_backoff_s
        self._clock = clock
        self._op_lock = threading.Lock()  # serialises orchestrator steps
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.trainer_metrics: dict[str, Any] = {}
        self._pool_status: dict[str, Any] = {}
        self._pool_status_t: float = 0.0
        self._pool_status_error: str | None = None
        self.started_at = self._clock()
        self.reconcile_errors = 0
        # per-replica timeline (filled by the poller): first seen active, first in-flight request, evicted
        self.replica_timeline: dict[str, dict[str, Any]] = {}
        self.health: dict[str, Any] = {}
        self.evictions: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- inputs
    def push_trainer_metrics(self, snap: dict[str, Any]) -> None:
        self.trainer_metrics.update(snap or {})

    def pool_status(self, force: bool = False) -> dict[str, Any]:
        now = self._clock()
        if force or now - self._pool_status_t > self._status_cache_s:
            try:
                self._pool_status = dict(self._pool_status_fn() or {})
                self._pool_status_error = None
            except Exception as e:  # noqa: BLE001
                self._pool_status_error = f"{type(e).__name__}: {e}"
            self._pool_status_t = now
        return self._pool_status

    def _note_timeline(self, pool: dict[str, Any], now: float) -> None:
        for a in pool.get("active") or []:
            rid = a.get("id")
            if not rid:
                continue
            rec = self.replica_timeline.setdefault(rid, {"first_active_t": now})
            if "first_inflight_t" not in rec and (a.get("inflight") or 0) > 0:
                rec["first_inflight_t"] = now
                rec["join_to_first_request_s"] = now - rec["first_active_t"]
        active_ids = {a.get("id") for a in pool.get("active") or []}
        for rid, rec in self.replica_timeline.items():
            if rid not in active_ids and "last_active_t" not in rec and "first_active_t" in rec:
                rec["last_active_t"] = now

    def poll_loop(self, interval_s: float = 5.0, health_every: int = 3, health_fn=None) -> None:
        """Background: refresh pool status (replica timeline) and run the liveness check; evict dead
        replicas under the op lock so the reconciler replaces them."""
        i = 0
        while not self._stop.is_set():
            now = self._clock()
            try:
                pool = self.pool_status(force=True)
                self._note_timeline(pool, now)
            except Exception:  # noqa: BLE001
                pass
            if health_fn is not None and i % health_every == 0:
                try:
                    h = health_fn()
                    self.health = h
                    dead = [d for d in (h.get("dead") or []) if d]
                    for rid in dead:
                        rec = self.replica_timeline.setdefault(rid, {})
                        if rec.get("evicted_t"):
                            continue
                        rec["dead_detected_t"] = now
                        with self._op_lock:
                            self.orch.evict_dead(rid, {"detected_at": now, "error": (h.get("errors") or {}).get(rid)})
                        rec["evicted_t"] = self._clock()
                        self.evictions.append({"replica_id": rid, "detected_t": now, "evicted_t": rec["evicted_t"], "error": (h.get("errors") or {}).get(rid)})
                        self._wake.set()
                except Exception as e:  # noqa: BLE001
                    self.health = {"error": f"{type(e).__name__}: {e}", "t": now}
            i += 1
            self._stop.wait(interval_s)

    # ---------------------------------------------------------------- facade for HTTP / ray
    def scale(self, standalone: int, release: bool | None = None) -> dict[str, Any]:
        res = self.orch.request(standalone, release=release)
        self._wake.set()
        return res

    def nodes(self) -> list[dict[str, Any]]:
        if self._nodes_fn is None:
            return []
        try:
            return list(self._nodes_fn() or [])
        except Exception as e:  # noqa: BLE001
            return [{"error": f"{type(e).__name__}: {e}"}]

    def status(self) -> dict[str, Any]:
        return {
            "orchestrator": self.orch.status(),
            "pool": self.pool_status(),
            "nodes": self.nodes(),
            "pool_status_error": self._pool_status_error,
            "trainer": self.trainer_metrics,
            "replica_timeline": self.replica_timeline,
            "health": self.health,
            "evictions": self.evictions,
            "uptime_s": self._clock() - self.started_at,
            "reconcile_errors": self.reconcile_errors,
        }

    def metrics_text(self) -> str:
        pool = self.pool_status()
        tm = self.trainer_metrics
        o = self.orch
        stats = pool.get("stats") or {}
        active = pool.get("active") or []
        parked = pool.get("parked") or []
        mq = tm.get("mq_size")
        if mq is None:
            mq = stats.get("monitor/queue/mq_queue_size")
        max_q = pool.get("max_queue_size")
        if max_q is None:
            max_q = stats.get("static/max_queue_size")
        rows: list[tuple[str, Any, dict | None, str | None, str | None]] = [
            ("rl_pool_active_standalone_replicas", len(active), None, "standalone replicas in the load balancer", "gauge"),
            ("rl_pool_sleeping_standalone_replicas", len(parked), None, "standalone replicas parked (out of LB, warm or asleep)", "gauge"),
            ("rl_pool_desired_standalone_replicas", o.desired, None, "last requested (clamped) standalone count", "gauge"),
            ("rl_pool_min_standalone_replicas", o.limits.min_standalone, None, None, "gauge"),
            ("rl_pool_max_standalone_replicas", o.limits.max_standalone, None, None, "gauge"),
            ("rl_pool_scaling_in_progress", o.in_progress or (o.desired is not None and o.desired != len(active)), None, "1 while converging", "gauge"),
            ("rl_pool_trainer_idle_ratio", tm.get("trainer_idle_ratio"), None, "trainer queue-wait share of step time, last sync cycle", "gauge"),
            ("rl_pool_trainer_wait_s", tm.get("trainer_wait_s"), None, "trainer seconds waiting on the message queue, last sync cycle", "gauge"),
            ("rl_pool_trainer_step_s", tm.get("trainer_step_s"), None, "trainer step wall seconds, last sync cycle", "gauge"),
            ("rl_pool_trainer_waiting_now", tm.get("trainer_waiting_now"), None, "1 while the trainer is blocked on the queue", "gauge"),
            ("rl_pool_trainer_wait_current_s", tm.get("trainer_wait_current_s"), None, None, "gauge"),
            ("rl_pool_trainer_weights_clean", tm.get("weights_clean"), None, "1 when actor weights equal the last synced version", "gauge"),
            ("rl_pool_mq_size", mq, None, "message queue size", "gauge"),
            ("rl_pool_max_queue_size", max_q, None, "message queue capacity", "gauge"),
            ("rl_pool_current_param_version", tm.get("current_param_version"), None, "trainer current_param_version", "gauge"),
            ("rl_pool_trainer_train_log_step", tm.get("train_log_step"), None, "step of the last aggregated training log (timing_s/step present); controllers key new samples on this", "gauge"),
            ("rl_pool_last_sync_wall_s", tm.get("last_sync_wall_s"), None, "wall seconds of the last regular weight sync", "gauge"),
            ("rl_pool_rollouter_idle_ratio", tm.get("rollouter_idle_ratio"), None, "rollouter paused share of the last version window", "gauge"),
            ("rl_pool_max_concurrent_samples", pool.get("max_concurrent_samples"), None, None, "gauge"),
            ("rl_pool_concurrent_samples_per_replica", pool.get("concurrent_samples_per_replica"), None, "admission cap per active replica (prompts)", "gauge"),
            ("rl_pool_rollout_resource_utilization", tm.get("rollout_util"), None, "time-weighted active_tasks / max_concurrent_samples over the last version window", "gauge"),
            ("rl_pool_response_length_mean", tm.get("response_length_mean"), None, None, "gauge"),
            ("rl_pool_num_turns_mean", tm.get("num_turns_mean"), None, None, "gauge"),
            ("rl_pool_score_mean", tm.get("score_mean"), None, None, "gauge"),
            ("rl_pool_dropped_stale", tm.get("dropped_stale"), None, None, "gauge"),
            ("rl_pool_aborted_ratio", tm.get("aborted_ratio"), None, None, "gauge"),
            ("rl_pool_partial_ratio", tm.get("partial_ratio"), None, None, "gauge"),
            ("rl_pool_active_tasks", stats.get("monitor/active_tasks_size"), None, "rollouter in-flight samples", "gauge"),
            ("rl_pool_lb_total_inflight", pool.get("lb_total_inflight"), None, None, "gauge"),
            ("rl_pool_reconcile_errors_total", self.reconcile_errors, None, None, "counter"),
            ("rl_pool_uptime_s", self._clock() - self.started_at, None, None, "gauge"),
        ]
        for k, v in o.ops_total.items():
            rows.append(("rl_pool_ops_total", v, {"op": k}, "orchestrator operations", "counter"))
        rows.append(("rl_pool_dead_replicas", len(self.health.get("dead") or []) if isinstance(self.health, dict) else 0, None, "replicas failing the liveness check at the last poll", "gauge"))
        rows.append(("rl_pool_evictions_total", len(self.evictions), None, None, "counter"))
        for a in active:
            rows.append(("rl_pool_replica_inflight_requests", a.get("inflight", 0), {"replica": a["id"], "address": a.get("address", "")}, "per-replica in-flight requests from the load balancer", "gauge"))
        versions = (tm.get("membership") or {}).get("versions") or {}
        for rid, v in versions.items():
            rows.append(("rl_pool_replica_param_version", v, {"replica": rid}, "param version last synced into the replica", "gauge"))
        return prom.render(rows)

    # ---------------------------------------------------------------- reconciler
    def reconcile_once(self) -> str | None:
        with self._op_lock:
            return self.orch.step()

    def reconcile_loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self._reconcile_idle_s)
            self._wake.clear()
            while not self._stop.is_set() and not self.orch.converged():
                try:
                    if self.reconcile_once() is None:
                        break
                except Exception:  # noqa: BLE001 - already recorded in orch.last_error
                    self.reconcile_errors += 1
                    self._stop.wait(self._error_backoff_s)
                    break

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()


class RLPoolControl:
    """Ray actor. Instantiate with ``ray.remote(RLPoolControl).options(name=..., num_cpus=0, max_concurrency=4)``."""

    def __init__(
        self,
        trainer,
        rollouter,
        min_standalone: int,
        max_standalone: int,
        port: int = 8787,
        release_on_scale_in: bool = False,
        join_timeout_s: float = 3600.0,
        join_poll_s: float = 5.0,
        scale_timeout_s: float = 1800.0,
        status_timeout_s: float = 30.0,
    ) -> None:
        import ray

        self._ray = ray
        self._rollouter = rollouter
        self._status_timeout_s = status_timeout_s
        backend = RayPoolBackend(trainer, rollouter, timeout_s=scale_timeout_s)
        orch = ScaleOrchestrator(
            backend,
            PoolLimits(min_standalone, max_standalone),
            release_on_scale_in=release_on_scale_in,
            join_timeout_s=join_timeout_s,
            join_poll_s=join_poll_s,
        )
        self.core = ControlCore(orch, self._pool_status, nodes_fn=self._ray_nodes)
        self._http = ControlHTTPServer(self.core, port=port).start()
        self._thread = threading.Thread(target=self.core.reconcile_loop, name="rl-pool-reconcile", daemon=True)
        self._thread.start()
        self._poller = threading.Thread(target=self.core.poll_loop, kwargs={"health_fn": backend.check_health}, name="rl-pool-poll", daemon=True)
        self._poller.start()
        print(f"[RLPoolControl] HTTP API on :{self._http.port} min={min_standalone} max={max_standalone}", flush=True)

    def _pool_status(self) -> dict[str, Any]:
        return self._ray.get(self._rollouter.get_pool_status.remote(), timeout=self._status_timeout_s)

    def _ray_nodes(self) -> list[dict[str, Any]]:
        """Alive Ray nodes with their GPU totals and (best effort) available GPUs; used by the
        Kubernetes-side actuator to wait for a new sampler pod before asking for a replica."""
        avail: dict[str, dict] = {}
        try:
            from ray._private import state as _state

            avail = dict(_state.available_resources_per_node() or {})
        except Exception:  # noqa: BLE001 - private API, best effort
            avail = {}
        out = []
        for n in self._ray.nodes():
            res = n.get("Resources") or {}
            if not n.get("Alive"):
                continue
            a = avail.get(n["NodeID"], {})
            out.append(
                {
                    "node_id": n["NodeID"],
                    "ip": n.get("NodeManagerAddress"),
                    "hostname": n.get("NodeManagerHostname"),
                    "gpu_total": res.get("GPU", 0),
                    # available_resources_per_node omits resources that are fully used, so a node that is
                    # present in the map but has no "GPU" key has 0 GPUs available (measured, Ray 2.56.1)
                    "gpu_available": (a.get("GPU", 0.0) if a else None),
                    "rollout_node": res.get("rollout_node", 0),
                    "trainer_node": res.get("trainer_node", 0),
                    "head": "node:__internal_head__" in res,
                }
            )
        return out

    # ray-callable
    def ready(self) -> dict[str, Any]:
        return {"port": self._http.port}

    def node_ip(self) -> str:
        return self._ray.util.get_node_ip_address()

    def push_trainer_metrics(self, snap: dict[str, Any]) -> None:
        self.core.push_trainer_metrics(snap)

    def scale(self, standalone: int, release: bool | None = None) -> dict[str, Any]:
        return self.core.scale(standalone, release)

    def status(self) -> dict[str, Any]:
        return self.core.status()

    def metrics_text(self) -> str:
        return self.core.metrics_text()

    def shutdown(self) -> None:
        self.core.stop()
        self._http.stop()
