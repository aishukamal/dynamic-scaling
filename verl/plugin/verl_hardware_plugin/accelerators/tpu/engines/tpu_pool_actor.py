# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Control endpoint of the TPU sampler pool (experiment, out of tree). Standard library + ray only.

A named Ray actor (``tpu_pool_control``) that holds the desired standalone replica count and the latest
state the trainer-side hook (``tpu_pool_control.PoolHook``) reports, and serves it over HTTP so the
Kubernetes-side controller (``rl_pool_autoscaler.controller.main``) can run unchanged:

    GET  /metrics   Prometheus text: rl_pool_* series (names the controller reads)
    GET  /status    JSON: replica sets, versions, trainer metrics, events, spawns
    POST /scale     {"standalone": N} -> desired count (clamped to [min, max]); the hook applies ONE move per sync

This module is imported by the actor process (the Ray head, no TPU), so it must not import torch/tpu modules.
"""

import http.server
import json
import threading
import time
from typing import Any


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


class PoolControlState:
    """State + HTTP server. Wrapped with ``ray.remote`` by the hook that creates it."""

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.min = int(cfg.get("min", 1))
        self.max = int(cfg.get("max", 8))
        self.lock = threading.Lock()
        self.desired = int(cfg.get("initial_active", 1))
        self.report: dict[str, Any] = {}
        self.trainer: dict[str, Any] = {}
        self.trainer_step: int | None = None
        self.history: list[tuple[int, dict]] = []
        self.window = int(cfg.get("window_steps", 2))
        self.mq_history: list[tuple[int, float]] = []
        self.events: list[dict] = []
        self.scale_requests: list[dict] = []
        self.t0 = time.time()
        self.port = int(cfg.get("port", 8787))
        # Product path (TPU-M3): scale-in releases (destroys) the replica instead of parking it when the controller asks
        # for it (POST /scale {"release": true}); the Kubernetes actuator reads the Ray node inventory from /status.
        self.release = bool(cfg.get("release_on_scale_in", False))
        self.rollout_group = str(cfg.get("rollout_group", "tpu-group-1"))
        self._serve()

    # ------------------------------------------------------------------ ray-facing API (hook side)
    def set_report(self, report: dict) -> None:
        with self.lock:
            self.report = dict(report)
            if report.get("sampleable") is not None and report.get("n_sync") is not None:
                self.mq_history.append((int(report["n_sync"]), float(report["sampleable"])))
                self.mq_history = self.mq_history[-8:]

    def set_trainer_metrics(self, step: int, metrics: dict) -> None:
        with self.lock:
            self.trainer_step = int(step)
            self.trainer = dict(metrics)
            self.history.append((int(step), dict(metrics)))
            self.history = self.history[-8:]

    def push_event(self, event: dict) -> None:
        with self.lock:
            self.events.append(event)
            if len(self.events) > 5000:
                self.events = self.events[-4000:]

    def get_desired(self) -> int:
        with self.lock:
            return self.desired

    def set_desired(self, n: int, source: str = "api", release: bool | None = None) -> dict:
        n = max(self.min, min(self.max, int(n)))
        with self.lock:
            prev = self.desired
            self.desired = n
            if release is not None:
                self.release = bool(release)
            rec = {"t": time.time(), "desired": n, "prev": prev, "source": source, "release": self.release}
            self.scale_requests.append(rec)
        return rec

    def get_release(self) -> bool:
        with self.lock:
            return self.release

    def _ray_nodes(self) -> list:
        """Alive Ray nodes with their TPU totals and (best effort) available TPUs, in the shape the GPU-M2 Kubernetes
        actuator reads (gpu_total / gpu_available / rollout_node): a sampler pod that joined Ray shows up as a rollout
        node with gpu_available == gpu_total until a replica's placement group takes its chips."""
        try:
            import ray
            from ray._private import state as _state

            avail = dict(_state.available_resources_per_node() or {})
            nodes = ray.nodes()
        except Exception as e:  # noqa: BLE001
            return [{"error": repr(e)[:200]}]
        out = []
        for n in nodes:
            if not n.get("Alive"):
                continue
            res = n.get("Resources") or {}
            a = avail.get(n["NodeID"], {})
            out.append({
                "node_id": n["NodeID"], "ip": n.get("NodeManagerAddress"), "hostname": n.get("NodeManagerHostname"),
                "gpu_total": res.get("TPU", 0), "gpu_available": (a.get("TPU", 0.0) if a else None),
                "rollout_node": res.get(self.rollout_group, 0), "trainer_node": res.get("tpu-group-0", 0),
                "head": "node:__internal_head__" in res,
            })
        return out

    def ping(self) -> float:
        return time.time()

    # ------------------------------------------------------------------ HTTP views
    def metrics_text(self) -> str:
        with self.lock:
            rep = dict(self.report)
            tr = dict(self.trainer)
            step = self.trainer_step
            desired = self.desired
        active = list(rep.get("active", []))
        parked = list(rep.get("parked", []))
        dead = list(rep.get("dead", []))
        step_s = tr.get("timing_s/step")
        gen_s = tr.get("timing_s/gen")
        idle_last = (float(gen_s) / float(step_s)) if step_s and gen_s is not None and float(step_s) > 0 else None
        # The separate_async trainer prefetches one batch, so wait and queue fill alternate with a 2-step period
        # (e.g. 38 s / 0 s). The controller reads the ratio over the last ``window_steps`` steps (sum of waits over
        # sum of step walls) and the queue fill as the mean sampleable count over the same window.
        with self.lock:
            hist = [m for _s, m in self.history[-self.window :]]
            mqh = [v for _s, v in self.mq_history[-self.window :]]
        num = sum(float(m.get("timing_s/gen", 0.0)) for m in hist)
        den = sum(float(m.get("timing_s/step", 0.0)) for m in hist)
        idle = (num / den) if den > 0 else idle_last
        mq_win = (sum(mqh) / len(mqh)) if mqh else rep.get("sampleable", 0)
        pending_repair = bool(rep.get("pending_repair", False))
        spawning = bool(rep.get("spawning", False))
        in_progress = (desired != len(active)) or pending_repair or spawning
        lines = []

        def g(name: str, value: Any, labels: dict | None = None, help_: str = "") -> None:
            if help_:
                lines.append(f"# HELP {name} {help_}")
                lines.append(f"# TYPE {name} gauge")
            lab = ""
            if labels:
                lab = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
            lines.append(f"{name}{lab} {_fmt(value)}")

        g("rl_pool_active_standalone_replicas", len(active), help_="active standalone replicas (in LB + membership)")
        g("rl_pool_sleeping_standalone_replicas", len(parked), help_="parked replicas (asleep level 2)")
        g("rl_pool_dead_replicas", len(dead), help_="replicas detected dead by the watchdog")
        g("rl_pool_desired_standalone_replicas", desired, help_="desired count (POST /scale)")
        g("rl_pool_scaling_in_progress", 1 if in_progress else 0, help_="desired != active, repair pending or spawn in flight")
        if step is not None:
            g("rl_pool_current_param_version", step, help_="trainer step of the last logged metrics")
            g("rl_pool_trainer_idle_ratio", idle, help_="sum(timing_s/gen)/sum(timing_s/step) over the last window_steps steps")
            g("rl_pool_trainer_idle_ratio_last", idle_last, help_="timing_s/gen / timing_s/step of the last logged step")
            g("rl_pool_trainer_wait_s", gen_s, help_="timing_s/gen of the last step")
            g("rl_pool_trainer_step_s", step_s, help_="timing_s/step of the last step")
            g("rl_pool_trainer_update_actor_s", tr.get("timing_s/update_actor"))
            g("rl_pool_trainer_update_weights_s", tr.get("timing_s/update_weights"))
            g("rl_pool_reward_mean", tr.get("critic/score/mean"))
            g("rl_pool_response_length_mean", tr.get("response_length/mean"))
            g("rl_pool_throughput_tok_s", tr.get("perf/throughput"))
            g("rl_pool_trainer_idle_ratio_ema", rep.get("idle_ema"))
        g("rl_pool_mq_size", mq_win, help_="mean sampleable prompt groups at sync time over the last window_steps syncs")
        g("rl_pool_mq_size_last", rep.get("sampleable", 0), help_="prompt groups already sampleable for the next step at the last sync")
        g("rl_pool_max_queue_size", rep.get("train_batch_size", 0), help_="prompt groups per step (train_batch_size)")
        g("rl_pool_last_sync_wall_s", rep.get("last_sync_wall_s"), help_="wall of the last Raiden sync (hook-measured)")
        g("rl_pool_last_sync_index", rep.get("n_sync"))
        g("rl_pool_last_sync_version", rep.get("last_sync_version"))
        g("rl_pool_lb_total_inflight", sum((rep.get("inflight") or {}).values()) if rep.get("inflight") else 0)
        # Slot utilization (TPU-M3): the controller's gate computes slot_util = rollout_resource_utilization x
        # max_concurrent_samples / (active x concurrent_samples_per_replica); exported so that it equals the time-weighted
        # decode-slot occupancy of the last step window (LB in-flight requests over active x max_num_seqs, clamped to 1).
        util = rep.get("util") or {}
        traj = rep.get("traj") or {}
        cap = rep.get("slot_cap") or 0
        g("rl_pool_rollout_resource_utilization", util.get("decode_slot_util"),
          help_="time-weighted mean of min(1, lb_inflight / (active x max_num_seqs)) over the last step window")
        g("rl_pool_max_concurrent_samples", len(active) * cap if cap else None, help_="active replicas x max_num_seqs")
        g("rl_pool_concurrent_samples_per_replica", cap or None, help_="vLLM max_num_seqs per replica (slot cap)")
        g("rl_pool_decode_slot_util", util.get("decode_slot_util"))
        g("rl_pool_share_slots_full", util.get("share_slots_full"), help_="share of the window with lb_inflight >= active x cap")
        g("rl_pool_lb_inflight_mean", util.get("lb_inflight_mean"))
        g("rl_pool_util_window_s", util.get("window_s"))
        g("rl_pool_active_tasks", traj.get("traj_inflight"), help_="trajectories in flight at the last step log (TQ running groups x n minus finished)")
        g("rl_pool_running_groups", traj.get("running_groups"))
        g("rl_pool_pending_groups", traj.get("pending_groups"))
        g("rl_pool_traj_slot_ratio", traj.get("traj_slot_ratio"), help_="trajectories in flight / (active x max_num_seqs), unclamped")
        for rank, n in (rep.get("inflight") or {}).items():
            g("rl_pool_replica_inflight_requests", n, {"replica": str(rank), "address": str((rep.get("addresses") or {}).get(str(rank), ""))})
        for rank, v in (rep.get("versions") or {}).items():
            g("rl_pool_replica_param_version", v, {"replica": str(rank)})
        g("rl_pool_uptime_s", time.time() - self.t0, help_="seconds since the control actor started")
        return "\n".join(lines) + "\n"

    def status(self) -> dict:
        nodes = self._ray_nodes()
        with self.lock:
            rep = dict(self.report)
            addr = rep.get("addresses") or {}
            pool = {k: [{"rank": r, "address": addr.get(str(r))} for r in (rep.get(k) or [])] for k in ("active", "parked", "dead")}
            return {
                "cfg": self.cfg,
                "desired": self.desired,
                "release": self.release,
                "trainer_step": self.trainer_step,
                "trainer": self.trainer,
                "report": rep,
                "pool": pool,
                "nodes": nodes,
                "scale_requests": self.scale_requests[-50:],
                "events": self.events[-400:],
                "uptime_s": time.time() - self.t0,
            }

    def _serve(self) -> None:
        state = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                return

            def _send(self, code: int, body: str, ctype: str = "text/plain; version=0.0.4"):
                data = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.startswith("/metrics"):
                    self._send(200, state.metrics_text())
                elif self.path.startswith("/status"):
                    self._send(200, json.dumps(state.status(), default=str), "application/json")
                elif self.path.startswith("/healthz"):
                    self._send(200, "ok")
                else:
                    self._send(404, "not found")

            def do_POST(self):
                if not self.path.startswith("/scale"):
                    self._send(404, "not found")
                    return
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n).decode() if n else "{}"
                try:
                    body = json.loads(raw or "{}")
                    target = int(body["standalone"])
                except (ValueError, KeyError, TypeError) as e:
                    self._send(400, json.dumps({"error": f"bad body: {e}"}), "application/json")
                    return
                rel = body.get("release")
                rec = state.set_desired(target, source="api", release=(bool(rel) if rel is not None else None))
                rep = state.report
                rec = dict(rec, active=len(rep.get("active", [])), parked=len(rep.get("parked", [])))
                self._send(200, json.dumps(rec), "application/json")

        self.httpd = http.server.ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self.httpd.daemon_threads = True
        th = threading.Thread(target=self.httpd.serve_forever, name="pool-http", daemon=True)
        th.start()
