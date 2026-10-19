# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU sampler-pool control hook (experiment driver, out of tree): the TPU-M1 late-join primitives driven by an
external controller instead of a fixed schedule.

Active only when ``VERL_TPU_POOL_CONTROL`` holds a JSON config; otherwise ``update_weights`` is the late-join spike
(``VERL_TPU_LATE_JOIN_SPIKE``) or plain ``update_raiden_weights``. Config keys::

    {"min": 1, "max": 8, "initial_active": 1, "log": "/data/tpu-m2/pool.jsonl", "port": 8787,
     "lb_poll_s": 2.0, "health_poll_s": 3.0, "health_timeout_s": 20.0, "probe": true,
     "spawn": false, "spawn_cache_dirs": {"1": "/tmp/torchinductor_cold_x"}}

Runs in the trainer process (the manager hook is called from ``on_step_end`` through
``CheckpointEngineManager.update_weights``). Membership changes happen ONLY in ``around()`` outside the Raiden
transfer, so no change races a sync; ONE policy move per sync (join or leave); a joiner is added to the load
balancer only after the sync that installed the current version; a replica found dead by the watchdog is cut from
the load balancer at once and from the membership at the next sync (a repair, not a policy move).

leave  = LB remove_servers -> abort_all_requests(reject) -> TPUWorker.sleep(level 2) -> manager.remove_replicas
join   = TPUWorker.wake_up(weights, kv_cache) -> manager.add_replicas -> [sync installs the version] ->
         LB add_servers + clear_sticky_cache -> probe (first token, greedy 16 vs the other replicas)
spawn  = (``spawn: true``) a NEW replica process set through ``RolloutReplica.init_standalone`` in a background
         thread, then the join path at the next sync (the fresh-pod product path, measured here as
         request -> process ready -> version installed -> first token).
join_on_ready = (``join_on_ready: true``) no sync-boundary wait: when the spawned engine is ready, a joiner-only
         Raiden push of the trainer's current clean weights (gated on an actor update in progress finishing, not
         on a whole version), membership add and LB add at once; a replica found dead is removed, destroyed and
         replaced at detection time. One lock serialises the push, the actor update and the regular sync.
"""

import asyncio
import gc
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

import ray

from verl_hardware_plugin.accelerators.tpu.engines import raiden_checkpoint_engine as _rce
from verl_hardware_plugin.accelerators.tpu.engines.raiden_checkpoint_engine import update_raiden_weights

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

CONTROL_ENV = "VERL_TPU_POOL_CONTROL"
SPIKE_ENV = "VERL_TPU_LATE_JOIN_SPIKE"
ACTOR_NAME = "tpu_pool_control"
PROBE_PROMPT = [785, 6722, 315, 9625, 374]
PROBE_TOKENS = 16
TRAINER_KEYS = (
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/adv",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "critic/score/mean",
    "response_length/mean",
    "perf/throughput",
    "perf/total_num_tokens",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _find_trainer(manager: Any) -> Any:
    """The trainer instance holding ``manager`` as ``standalone_checkpoint_manager`` (no back-reference exists)."""
    for d in gc.get_referrers(manager):
        if isinstance(d, dict) and d.get("standalone_checkpoint_manager") is manager:
            for obj in gc.get_referrers(d):
                if getattr(obj, "__dict__", None) is d and hasattr(obj, "standalone_server_manager"):
                    return obj
    raise RuntimeError("pool control: no trainer holds this manager as standalone_checkpoint_manager")


def _head_node_ip() -> Optional[str]:
    for node in ray.nodes():
        if node.get("Alive") and "node:__internal_head__" in node.get("Resources", {}):
            return node.get("NodeManagerAddress")
    return None


class PoolHook:
    def __init__(self, manager: Any, cfg: dict):
        self.manager = manager
        self.cfg = cfg
        self.min = int(cfg.get("min", 1))
        self.max = int(cfg.get("max", 8))
        self.log_path = cfg.get("log", "/data/tpu-m2/pool.jsonl")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.trainer = _find_trainer(manager)
        self.server_manager = self.trainer.standalone_server_manager
        self.lb = self.server_manager.global_load_balancer
        self.all: dict[int, Any] = {r.replica_rank: r for r in manager.replicas}
        self.active: list[int] = sorted(self.all)
        self.parked: list[int] = []
        self.dead: dict[int, dict] = {}
        self.dead_in_manager: set[int] = set()
        self.versions: dict[int, int] = {}
        self.spawn_ready: list[int] = []
        self.destroyed: list[int] = []
        self.spawns: dict[int, dict] = {}
        self.spawning = False
        self.next_rank = (max(self.all) + 1) if self.all else 0
        self.n_sync = 0
        self.last_sync_wall: Optional[float] = None
        self.last_sync_version: Optional[int] = None
        self.idle_ema: Optional[float] = None
        self.initial_active = int(cfg.get("initial_active", len(self.active)))
        self.initial_done = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # join_on_ready (TPU-M3 dyn3): one lock serialises everything that reads or writes the trainer's weights or the
        # Raiden membership: the regular sync (around), the actor update (wrapped _update_actor), a joiner-only push
        # (_join_on_ready) and a detection-time repair (_repair_dead_now). _actor_updated_pending marks "update_actor of
        # this step finished, its sync has not run yet", which is the only case where the clean version is last+1.
        self._sync_lock = threading.Lock()
        self._spawn_gate = threading.Lock()  # one spawn decision at a time (early-spawn poller vs detection-time repair)
        self._actor_busy = False
        self._actor_updated_pending = False
        self._trainer_blocked_s = 0.0
        self.join_on_ready = bool(cfg.get("join_on_ready", False))
        self._inflight: dict[str, int] = {}
        self._logger_wrapped = False
        # Slot utilization (TPU-M3): per-replica slot cap = vLLM max_num_seqs (the admission cap the V1 path lacks); the LB
        # sampler records (mono, lb_total_inflight, n_active) and the step tap folds them into time-weighted means per step.
        try:
            rc = self.trainer.config.actor_rollout_ref.rollout
            self.slot_cap = int(rc.max_num_seqs)
            self.rollout_n = int(rc.n)
        except Exception:  # noqa: BLE001
            self.slot_cap, self.rollout_n = 0, 1
        self._occ: list[tuple[float, int, int]] = []
        self._occ_lock = threading.Lock()
        self._occ_last_cut = time.perf_counter()
        self.util: dict[str, Any] = {}
        self.traj: dict[str, Any] = {}
        try:
            self.train_batch_size = int(self.trainer.config.data.train_batch_size)
        except Exception:  # noqa: BLE001
            self.train_batch_size = 0
        self.ctrl = self._create_actor()
        self.log("init", replicas=sorted(self.all), addresses=self.addresses(), cfg=cfg, actor=str(self.ctrl),
                 actor_host=self.actor_host, port=int(cfg.get("port", 8787)))
        if self.join_on_ready:
            self._wrap_update_actor()
        threading.Thread(target=self._lb_sampler, name="pool-lb-sampler", daemon=True).start()
        threading.Thread(target=self._watchdog, name="pool-watchdog", daemon=True).start()
        if self.cfg.get("spawn", False) and self.cfg.get("early_spawn", False):
            # TPU-M3 long arms: create the replica process as soon as the controller asks (engine init does not touch the
            # trainer); the MEMBERSHIP change (Raiden registration, LB add) still happens only at the next sync boundary.
            threading.Thread(target=self._early_spawn_poller, name="pool-early-spawn", daemon=True).start()

    # ------------------------------------------------------------------ plumbing
    def _create_actor(self) -> Any:
        from verl_hardware_plugin.accelerators.tpu.engines.tpu_pool_actor import PoolControlState

        opts: dict[str, Any] = {"name": ACTOR_NAME, "get_if_exists": True, "num_cpus": 0, "max_concurrency": 32}
        head_ip = self.cfg.get("actor_host") or _head_node_ip()
        if head_ip:
            # Hard pin to the head through its node:<ip> resource (a soft NodeAffinity landed the actor on the
            # TaskRunner's node in the first run); the HTTP endpoint is then at <head pod ip>:port.
            opts["resources"] = {f"node:{head_ip}": 0.001}
        self.actor_host = head_ip
        actor_cfg = {
            "min": self.min,
            "max": self.max,
            "initial_active": self.initial_active,
            "port": int(self.cfg.get("port", 8787)),
            "train_batch_size": self.train_batch_size,
        }
        actor = ray.remote(PoolControlState).options(**opts).remote(actor_cfg)
        ray.get(actor.ping.remote(), timeout=120)
        return actor

    def addresses(self) -> dict[str, str]:
        return {str(k): str(r.server_address) for k, r in self.all.items()}

    def log(self, kind: str, **kw: Any) -> None:
        rec = {"t": _utc(), "mono": time.perf_counter(), "kind": kind, "sync": self.n_sync, **kw}
        line = json.dumps(rec, default=str)
        try:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
        logger.info(f"[POOL] {line}")
        try:
            self.ctrl.push_event.remote(rec)
        except Exception:  # noqa: BLE001
            pass

    def _report(self, **extra: Any) -> None:
        rep = {
            "t": _utc(),
            "active": list(self.active),
            "parked": list(self.parked),
            "dead": sorted(self.dead),
            "pending_repair": bool(set(self.dead) - self.dead_in_manager),
            "spawning": self.spawning,
            "spawn_ready": list(self.spawn_ready),
            "destroyed": list(self.destroyed),
            "versions": {str(k): v for k, v in self.versions.items()},
            "addresses": self.addresses(),
            "inflight": dict(self._inflight),
            "n_sync": self.n_sync,
            "last_sync_wall_s": self.last_sync_wall,
            "last_sync_version": self.last_sync_version,
            "train_batch_size": self.train_batch_size,
            "idle_ema": self.idle_ema,
            "slot_cap": self.slot_cap,
            "rollout_n": self.rollout_n,
            "util": dict(self.util),
            "traj": dict(self.traj),
            **extra,
        }
        try:
            self.ctrl.set_report.remote(rep)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[POOL] report failed: {e}")

    def _lb_sampler(self) -> None:
        poll = float(self.cfg.get("lb_poll_s", 2.0))
        while not self._stop.is_set():
            try:
                st = ray.get(self.lb.get_status.remote(), timeout=10)
                addr_to_rank = {v: k for k, v in self.addresses().items()}
                servers = {str(addr_to_rank.get(a, a)): n for a, n in st.get("servers", {}).items()}
                self._inflight = servers
                lb_total = sum(int(v) for v in servers.values())
                with self._occ_lock:
                    self._occ.append((time.perf_counter(), lb_total, len(self.active)))
                    if len(self._occ) > 20000:
                        self._occ = self._occ[-10000:]
                rec = {"t": _utc(), "mono": time.perf_counter(), "kind": "lb", "inflight": servers, "lb_total": lb_total,
                       "n_active": len(self.active), "cap": self.slot_cap}
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[POOL] lb sampler: {e}")
            time.sleep(poll)

    def _watchdog(self) -> None:
        """Marks an ACTIVE replica dead when a no-op worker RPC fails or times out; cuts it from the LB at once."""
        poll = float(self.cfg.get("health_poll_s", 3.0))
        timeout = float(self.cfg.get("health_timeout_s", 20.0))
        needed = int(self.cfg.get("health_failures", 2))
        failures: dict[int, int] = {}
        first_fail: dict[int, tuple[str, float]] = {}
        while not self._stop.is_set():
            for rank in list(self.active):
                if rank in self.dead:
                    continue
                r = self.all.get(rank)
                if r is None:
                    continue
                t0 = time.perf_counter()
                try:
                    # WorkerBase.check_health is a no-op function on every vLLM worker (TPUWorker.is_sleeping is a property
                    # in this image, not callable through run_method). The RPC crosses server actor -> EngineCore -> workers.
                    ray.get(r.server_handle.collective_rpc.remote(method="check_health"), timeout=timeout)
                    failures.pop(rank, None)
                    first_fail.pop(rank, None)
                except Exception as e:  # noqa: BLE001
                    failures[rank] = failures.get(rank, 0) + 1
                    first_fail.setdefault(rank, (_utc(), t0))
                    self.log("health_probe_failed", rank=rank, n=failures[rank], error=repr(e)[:300])
                    if failures[rank] < needed:
                        continue
                    # Only a TIMEOUT is ambiguous (slow engine vs dead engine). Any other error (EngineDeadError after a
                    # SIGKILLed vLLM worker, an actor-died error) is a dead engine: the chip-holding RayWorkerWrapper actors
                    # belong to vLLM's executor, not to ``r.workers``, so their death does not show in the actor-state check
                    # (tpum3-dyn3 failure act 22:21:44Z: EngineDeadError from 22:22:19Z, every listed actor ALIVE, the dead
                    # path only opened after hung_after_s).
                    is_timeout = isinstance(e, ray.exceptions.GetTimeoutError) or "GetTimeoutError" in repr(e)
                    states = self._actor_states(r) if (self.join_on_ready and is_timeout) else None
                    if states is not None and states and all(v == "ALIVE" for v in states.values()):
                        # Every server/worker actor is alive: a slow engine (first-bucket compile), not a dead one. With the
                        # detection-time destroy a false positive would kill a healthy replica (tpum3-dyn3 19:13:23Z), so
                        # keep probing without touching the LB until the probes have failed for hung_after_s.
                        since = time.perf_counter() - first_fail[rank][1]
                        if since < float(self.cfg.get("hung_after_s", 300.0)):
                            self.log("health_probe_slow", rank=rank, n=failures[rank], since_first_fail_s=since,
                                     actor_states=states)
                            continue
                    t_detect = time.perf_counter()
                    info = {"t_detect": _utc(), "mono_detect": t_detect, "probe_s": t_detect - t0, "error": repr(e)[:400],
                            "t_first_fail": first_fail[rank][0], "first_fail_to_detect_s": t_detect - first_fail[rank][1],
                            "actor_states": states}
                    try:
                        ray.get(self.lb.remove_servers.remote([r.server_address]), timeout=30)
                        info["lb_removed_s"] = time.perf_counter() - t_detect
                    except Exception as e2:  # noqa: BLE001
                        info["lb_remove_error"] = repr(e2)[:200]
                    with self._lock:
                        self.dead[rank] = info
                    self.log("dead_detected", rank=rank, **info, active=list(self.active))
                    self._report()
                    if self.join_on_ready:
                        try:
                            self._repair_dead_now(rank)
                        except Exception as e3:  # noqa: BLE001
                            self.log("repair_error", rank=rank, error=repr(e3)[:400])
            time.sleep(poll)

    # ------------------------------------------------------------------ join_on_ready (no sync-boundary wait)
    def _wrap_update_actor(self) -> None:
        """Wraps the trainer's actor update so a joiner-only push never overlaps a weight update: the push waits for
        an update in progress (wait-for-clean-weights), and an update waits for a push in progress (recorded)."""
        orig = self.trainer._update_actor
        hook = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            with hook._sync_lock:
                waited = time.perf_counter() - t0
                if waited > 0.5:
                    hook._trainer_blocked_s += waited
                    hook.log("trainer_blocked_by_join", waited_s=waited, total_blocked_s=hook._trainer_blocked_s)
                hook._actor_busy = True
                try:
                    return orig(*args, **kwargs)
                finally:
                    hook._actor_busy = False
                    hook._actor_updated_pending = True

        self.trainer._update_actor = wrapped
        self.log("update_actor_wrapped")

    def _clean_version(self) -> int:
        """Version of the trainer's weights while _sync_lock is held and no update is running."""
        return int(self.last_sync_version or 0) + (1 if self._actor_updated_pending else 0)

    def _join_on_ready(self, rank: int, replica: Any, sp: dict) -> None:
        """Spawned replica -> serving without waiting for the next sync: joiner-only Raiden push of the current clean
        weights (manager.replicas swapped to [replica] for the push, unique transfer req_id), membership add, LB add."""
        rec: dict[str, Any] = {"rank": rank, "actor_busy_at_request": self._actor_busy,
                               "t_request_push": _utc()}
        t_req = time.perf_counter()
        with self._sync_lock:
            t_lock = time.perf_counter()
            rec["wait_clean_weights_s"] = t_lock - t_req
            with self._lock:
                already = rank in self.active or rank not in self.spawn_ready or rank not in self.all
            if already:
                self.log("join_on_ready_skipped", **rec, reason="already joined at a sync or gone", active=list(self.active))
                return
            version = self._clean_version()
            rec["version"] = version
            rec["actor_updated_pending"] = self._actor_updated_pending
            saved = list(self.manager.replicas)
            self.manager.replicas = [replica]
            _rce.REQ_ID_SUFFIX = f"_join{rank}_{int(time.time())}"
            try:
                metrics = asyncio.run(update_raiden_weights(self.manager, global_steps=version))
                rec["sync_error"] = None
            except Exception as e:  # noqa: BLE001
                metrics = {}
                rec["sync_error"] = repr(e)[:600]
            finally:
                _rce.REQ_ID_SUFFIX = ""
                self.manager.replicas = saved
            t_sync = time.perf_counter()
            rec["joiner_sync_s"] = t_sync - t_lock
            rec["phases"] = {k.split("/")[-1]: v for k, v in (metrics or {}).items() if k.startswith("timing_s/tpu-sync/")}
            if rec["sync_error"]:
                self.log("join_on_ready_error", **rec, active=list(self.active))
                return
            self.manager.add_replicas([replica])
            try:
                ray.get(self.lb.add_servers.remote({replica.server_address: replica.server_handle}), timeout=60)
                rec["cleared"] = ray.get(self.lb.clear_sticky_cache.remote(), timeout=60)
            except Exception as e:  # noqa: BLE001
                rec["lb_add_error"] = repr(e)[:300]
            t_lb = time.perf_counter()
            rec["lb_add_s"] = t_lb - t_sync
            self.versions[rank] = version
            with self._lock:
                if rank in self.spawn_ready:
                    self.spawn_ready.remove(rank)
                self.active.append(rank)
        placed = sp.get("mono_placed", sp.get("mono_start"))
        sp.update(joined_on_ready=True, t_serving=_utc(), version=version,
                  pod_ray_s=(placed - sp["mono_start"]) if placed is not None else None,
                  engine_init_s=(sp["mono_ready"] - placed) if placed is not None else None,
                  wait_clean_weights_s=rec["wait_clean_weights_s"], joiner_sync_s=rec["joiner_sync_s"], lb_add_s=rec["lb_add_s"],
                  ready_to_serving_s=t_lb - sp["mono_ready"],
                  req_to_serving_s=(t_lb - sp["t_request"]) if sp.get("t_request") else None)
        self.log("join_on_ready", **rec, **{k: v for k, v in sp.items() if not k.startswith("mono") and k not in rec},
                 active=list(self.active), manager_replicas=[x.replica_rank for x in self.manager.replicas])
        self._report()

    def _repair_dead_now(self, rank: int) -> None:
        """Detection-time repair (join_on_ready): membership removal + destroy as soon as the watchdog marks a replica
        dead, then a replacement spawn if the controller's desired count exceeds what the pool has."""
        t0 = time.perf_counter()
        with self._sync_lock:
            waited = time.perf_counter() - t0
            r = self.all.get(rank)
            if rank in self.dead_in_manager or r is None:
                return
            if any(x is r for x in self.manager.replicas):
                self.manager.remove_replicas([r])
            self.dead_in_manager.add(rank)
            with self._lock:
                if rank in self.active:
                    self.active.remove(rank)
            self.log("dead_removed", rank=rank, source="watchdog", waited_lock_s=waited, active=list(self.active),
                     manager_replicas=[x.replica_rank for x in self.manager.replicas])
            if self.cfg.get("destroy_dead", True):
                asyncio.run(self.destroy(rank, "dead"))
        try:
            desired = int(ray.get(self.ctrl.get_desired.remote(), timeout=10))
        except Exception as e:  # noqa: BLE001
            self.log("desired_error", error=repr(e)[:200], source="repair")
            return
        with self._spawn_gate:  # the early-spawn poller takes the same gate, so only one of them starts the replacement
            with self._lock:
                have = len(self.active) + len(self.parked) + len(self.spawn_ready)
            if desired > have and self.cfg.get("spawn", False) and not self.spawning:
                if self.next_rank < int(self.cfg.get("max_rank", 8)):
                    self.log("replacement_spawn", desired=desired, have=have, dead_rank=rank)
                    self.start_spawn(t_request=t0)
            else:
                self.log("spawn_refused", reason="max_rank reached", next_rank=self.next_rank)

    def _actor_states(self, r: Any) -> Optional[dict[str, str]]:
        """Ray actor state (ALIVE/DEAD/...) of the replica's server and worker actors, from the GCS (no RPC into the
        engine); None when the state API is unavailable. A SIGKILLed worker process shows DEAD within seconds."""
        try:
            from ray.util.state import get_actor
        except Exception:  # noqa: BLE001
            return None
        out: dict[str, str] = {}
        try:
            for kind, handles in (("server", getattr(r, "servers", None) or []), ("worker", getattr(r, "workers", None) or [])):
                for i, h in enumerate(handles):
                    st = get_actor(id=h._actor_id.hex(), timeout=10)
                    out[f"{kind}{i}"] = str(getattr(st, "state", None) or "MISSING") if st is not None else "MISSING"
        except Exception as e:  # noqa: BLE001
            self.log("actor_state_error", rank=getattr(r, "replica_rank", None), error=repr(e)[:200])
            return None
        return out

    def _sampleable(self) -> Optional[int]:
        rb = getattr(self.trainer, "replay_buffer", None)
        fn = getattr(rb, "get_sampleable_count", None)
        if fn is None:
            return None
        try:
            return int(fn(int(self.trainer.global_steps) + 1, "train"))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[POOL] sampleable: {e}")
            return None

    def _wrap_logger_once(self) -> None:
        if self._logger_wrapped:
            return
        lg = getattr(self.trainer, "logger", None)
        if lg is None:
            return
        orig = lg.log
        hook = self

        def log(*args, **kwargs):
            data = kwargs.get("data", args[0] if args else None)
            step = kwargs.get("step", args[1] if len(args) > 1 else None)
            try:
                hook.on_trainer_log(data, step)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[POOL] trainer log tap: {e}")
            return orig(*args, **kwargs)

        lg.log = log
        self._logger_wrapped = True

    def on_trainer_log(self, data: Any, step: Any) -> None:
        if not isinstance(data, dict) or step is None:
            return
        picked = {k: data[k] for k in TRAINER_KEYS if k in data}
        picked.update({k: v for k, v in data.items() if k.startswith("timing_s/tpu-sync/")})
        if not picked:
            return
        step_s, gen_s = picked.get("timing_s/step"), picked.get("timing_s/gen")
        if step_s and gen_s is not None and float(step_s) > 0:
            idle = float(gen_s) / float(step_s)
            self.idle_ema = idle if self.idle_ema is None else 0.5 * self.idle_ema + 0.5 * idle
        self.util = self._fold_occupancy()
        self.traj = self._traj_inflight()
        self.ctrl.set_trainer_metrics.remote(int(step), picked)
        self.log("step", step=int(step), active=list(self.active), n_active=len(self.active), metrics=picked,
                 util=self.util, traj=self.traj)
        self._report()

    def _fold_occupancy(self) -> dict:
        """Time-weighted means of the LB occupancy samples since the previous step log (= one step window).

        decode_slot_util = mean over time of min(1, lb_total_inflight / (n_active * slot_cap)): the share of the active replicas'
        vLLM slots (max_num_seqs) occupied by generate requests (requests queued inside vLLM count, so this saturates at 1 while a
        batch is backed up and falls in the tail). Also the raw mean in-flight and the mean active count over the window.
        """
        now = time.perf_counter()
        with self._occ_lock:
            cut = self._occ_last_cut
            samples = [x for x in self._occ if x[0] >= cut]
            self._occ_last_cut = now
            self._occ = [x for x in self._occ if x[0] >= cut]
        if len(samples) < 2 or not self.slot_cap:
            return {"window_s": now - cut, "samples": len(samples)}
        num_u = num_in = num_act = 0.0
        den = 0.0
        full = 0.0
        for (t0, lb, na), (t1, _lb1, _na1) in zip(samples, samples[1:], strict=False):
            dt = max(0.0, t1 - t0)
            cap = max(1, na) * self.slot_cap
            u = min(1.0, lb / cap)
            num_u += u * dt
            num_in += lb * dt
            num_act += na * dt
            full += dt if lb >= cap else 0.0
            den += dt
        if den <= 0:
            return {"window_s": now - cut, "samples": len(samples)}
        return {"window_s": now - cut, "samples": len(samples), "decode_slot_util": num_u / den,
                "lb_inflight_mean": num_in / den, "n_active_mean": num_act / den, "share_slots_full": full / den,
                "cap_per_replica": self.slot_cap}

    def _traj_inflight(self) -> dict:
        """Trajectories in flight right now (agent loops started and not finished, incl. time in the sandbox), from the
        TransferQueue metadata: running prompt groups x rollout.n minus their finished trajectories. Called from the trainer
        thread only (the TQ client is used by the replay buffer on that thread)."""
        try:
            import transfer_queue as tq

            data = tq.kv_list()
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)[:120]}
        part = (data or {}).get("train") or {}
        running = {k for k, tag in part.items() if tag.get("is_prompt") and tag.get("status") == "running"}
        pending = sum(1 for tag in part.values() if tag.get("is_prompt") and tag.get("status") == "pending")
        done: dict[str, int] = {}
        for k, tag in part.items():
            if tag.get("is_prompt"):
                continue
            uid = k.rsplit("_", 2)[0]
            if uid in running:
                done[uid] = done.get(uid, 0) + 1
        inflight = sum(max(0, self.rollout_n - done.get(u, 0)) for u in running)
        cap = max(1, len(self.active)) * max(1, self.slot_cap)
        return {"running_groups": len(running), "pending_groups": pending, "traj_inflight": inflight,
                "traj_slot_ratio": inflight / cap}

    # ------------------------------------------------------------------ primitives (TPU-M1)
    async def leave(self, rank: int, why: str) -> None:
        r = self.all[rank]
        t0 = time.perf_counter()
        await self.lb.remove_servers.remote([r.server_address])
        t1 = time.perf_counter()
        aborted = await r.server_handle.abort_all_requests.remote(reject_request=True)
        t2 = time.perf_counter()
        slept = await r.server_handle.collective_rpc.remote(method="sleep", kwargs={"level": 2})
        t3 = time.perf_counter()
        self.manager.remove_replicas([r])
        if rank in self.active:
            self.active.remove(rank)
        if rank not in self.parked:
            self.parked.append(rank)
        self.log(
            "leave", rank=rank, why=why, lb_remove_s=t1 - t0, abort_s=t2 - t1, sleep_s=t3 - t2, total_s=t3 - t0,
            aborted=aborted, sleep_info=slept, manager_replicas=[x.replica_rank for x in self.manager.replicas],
            active=list(self.active), parked=list(self.parked),
        )

    async def destroy(self, rank: int, why: str) -> dict:
        """Product-path removal (TPU-M3): LB cut -> abort (unless already dead) -> out of the Raiden membership -> kill
        the replica's server and worker actors -> remove its placement groups, so the host's chips are free for a
        replacement spawn or for the Kubernetes actuator to delete the pod. Nothing is parked."""
        r = self.all.get(rank)
        rec: dict[str, Any] = {"rank": rank, "why": why}
        if r is None:
            self.log("destroy", **rec, error="unknown rank")
            return rec
        t0 = time.perf_counter()
        try:
            await self.lb.remove_servers.remote([r.server_address])
        except Exception as e:  # noqa: BLE001
            rec["lb_remove_error"] = repr(e)[:200]
        t1 = time.perf_counter()
        if rank not in self.dead:
            try:
                rec["aborted"] = await asyncio.wait_for(r.server_handle.abort_all_requests.remote(reject_request=True), timeout=30)
            except Exception as e:  # noqa: BLE001
                rec["abort_error"] = repr(e)[:200]
        t2 = time.perf_counter()
        try:
            if any(x is r for x in self.manager.replicas):
                self.manager.remove_replicas([r])
        except Exception as e:  # noqa: BLE001
            rec["manager_remove_error"] = repr(e)[:200]
        killed = 0
        for h in list(getattr(r, "servers", None) or []) + list(getattr(r, "workers", None) or []):
            try:
                ray.kill(h, no_restart=True)
                killed += 1
            except Exception:  # noqa: BLE001
                pass
        removed = 0
        for pg in list(getattr(getattr(r, "resource_pool", None), "pgs", None) or []):
            try:
                ray.util.remove_placement_group(pg)
                removed += 1
            except Exception:  # noqa: BLE001
                pass
        sm = self.server_manager
        for attr, item in (("rollout_replicas", r), ("server_handles", getattr(r, "server_handle", None)),
                           ("server_addresses", getattr(r, "server_address", None))):
            try:
                getattr(sm, attr).remove(item)
            except (ValueError, AttributeError):
                pass
        with self._lock:
            for coll in (self.active, self.parked, self.spawn_ready):
                if rank in coll:
                    coll.remove(rank)
            self.all.pop(rank, None)
            self.dead_in_manager.add(rank)
            self.destroyed.append(rank)
        t3 = time.perf_counter()
        rec.update(lb_remove_s=t1 - t0, abort_s=t2 - t1, kill_s=t3 - t2, total_s=t3 - t0, killed_actors=killed,
                   pgs_removed=removed, active=list(self.active),
                   manager_replicas=[x.replica_rank for x in self.manager.replicas])
        self.log("destroy", **rec)
        self._report()
        return rec

    async def join_before_sync(self, rank: int, why: str) -> dict:
        r = self.all[rank]
        rec: dict[str, Any] = {"rank": rank, "why": why}
        t0 = time.perf_counter()
        if rank in self.parked:
            try:
                rec["wake"] = await r.server_handle.collective_rpc.remote(
                    method="wake_up", kwargs={"tags": ["weights", "kv_cache"]}
                )
            except Exception as e:  # noqa: BLE001
                rec["wake_error"] = repr(e)[:300]
            rec["wake_s"] = time.perf_counter() - t0
            self.parked.remove(rank)
        if rank in self.spawn_ready:
            self.spawn_ready.remove(rank)
        if any(x is r for x in self.manager.replicas):
            # a spawned replica is already in the server manager's list (start_spawn appends it); adding it again made the
            # Raiden transfer address the unit twice (TPU-M2 spawns and TPU-M3 join 1: units [0, 1, 1])
            rec["already_in_manager"] = True
        else:
            self.manager.add_replicas([r])
        self.active.append(rank)
        rec["manager_replicas"] = [x.replica_rank for x in self.manager.replicas]
        self.log("join_before_sync", **rec, active=list(self.active), parked=list(self.parked))
        return rec

    async def _probe(self, r: Any, request_id: str, max_tokens: int) -> tuple[float, list[int]]:
        t0 = time.perf_counter()
        out = await r.server_handle.generate.remote(
            prompt_ids=list(PROBE_PROMPT),
            sampling_params={"temperature": 0.0, "max_tokens": max_tokens, "logprobs": True},
            request_id=request_id,
        )
        return time.perf_counter() - t0, list(getattr(out, "token_ids", []) or [])

    async def join_after_sync(self, rank: int, global_steps: Optional[int]) -> dict:
        r = self.all[rank]
        t0 = time.perf_counter()
        await self.lb.add_servers.remote({r.server_address: r.server_handle})
        cleared = await self.lb.clear_sticky_cache.remote()
        t1 = time.perf_counter()
        rec: dict[str, Any] = {"rank": rank, "lb_add_s": t1 - t0, "cleared": cleared, "version": global_steps}
        self.versions[rank] = int(global_steps or 0)
        if self.cfg.get("probe", True):
            try:
                ft_s, ft_ids = await self._probe(r, f"pool-ft-{rank}-{self.n_sync}", 1)
                rec.update(first_token_s=ft_s, first_token_ids=ft_ids)
                greedy = {}
                # The greedy comparison queues behind the other replicas' in-flight requests (TPU-M3 join 1: 220 s on a replica
                # with 355 in flight, stalling the trainer); run it only when every other replica's queue is short.
                max_load = max([int(v) for k, v in self._inflight.items() if str(k) != str(rank)] or [0])
                probe_max_load = int(self.cfg.get("probe_max_load", 32))
                if max_load > probe_max_load:
                    rec["greedy_skipped"] = f"other replicas' in-flight {max_load} > {probe_max_load}"
                else:
                    seen_ids = set()
                    for x in self.manager.replicas:
                        if x.replica_rank in self.dead or id(x) in seen_ids:
                            continue
                        seen_ids.add(id(x))
                        s, ids = await self._probe(x, f"pool-greedy-{x.replica_rank}-{self.n_sync}", PROBE_TOKENS)
                        greedy[x.replica_rank] = {"s": s, "ids": ids}
                    others = [v["ids"] for k, v in greedy.items() if k != rank]
                    rec.update(greedy=greedy, greedy_equal_to_others=[v == greedy[rank]["ids"] for v in others])
            except Exception as e:  # noqa: BLE001
                rec["probe_error"] = repr(e)[:300]
        self.log("join_after_sync", **rec, active=list(self.active))
        return rec

    # ------------------------------------------------------------------ cold spawn (init_standalone)
    def start_spawn(self, t_request: Optional[float]) -> None:
        rank = self.next_rank
        self.next_rank += 1
        self.spawning = True
        rec: dict[str, Any] = {
            "rank": rank,
            "t_request": t_request,
            "t_start": _utc(),
            "mono_start": time.perf_counter(),
            "req_to_start_s": (time.perf_counter() - t_request) if t_request else None,
        }
        self.spawns[rank] = rec
        cache_dir = (self.cfg.get("spawn_cache_dirs") or {}).get(str(rank))
        if cache_dir:
            os.environ[f"VERL_TPU_POOL_WORKER_ENV_{rank}"] = json.dumps({"TORCHINDUCTOR_CACHE_DIR": cache_dir})
            rec["cache_dir"] = cache_dir
        self.log("spawn_start", **{k: v for k, v in rec.items() if k != "mono_start"})

        def run() -> None:
            try:
                sm = self.server_manager
                replica = sm.rollout_replica_class(
                    replica_rank=rank,
                    config=sm.rollout_config,
                    model_config=sm.model_config,
                    gpus_per_node=sm.rollout_config.n_gpus_per_node,
                )
                # TPU-M3: a replacement spawn right after a destroy can see Ray's TPU count before the removed placement
                # group's bundles are released ("Total available TPU 0 is less than total desired TPU 4"); retry briefly.
                attempts = int(self.cfg.get("spawn_retries", 12))
                for attempt in range(attempts + 1):
                    try:
                        rec["mono_placed"] = time.perf_counter()  # start of the attempt that ends up succeeding
                        rec["attempts"] = attempt + 1
                        asyncio.run(replica.init_standalone())
                        break
                    except ValueError as e:
                        if "available TPU" in str(e) and attempt < attempts:
                            self.log("spawn_retry", rank=rank, attempt=attempt + 1, error=repr(e)[:200])
                            time.sleep(float(self.cfg.get("spawn_retry_s", 10.0)))
                            replica = sm.rollout_replica_class(replica_rank=rank, config=sm.rollout_config,
                                                               model_config=sm.model_config, gpus_per_node=sm.rollout_config.n_gpus_per_node)
                            continue
                        raise
                rec["mono_ready"] = time.perf_counter()
                rec["t_ready"] = _utc()
                rec["init_standalone_s"] = rec["mono_ready"] - rec["mono_start"]
                rec["address"] = replica.server_address
                with self._lock:
                    self.all[rank] = replica
                    self.spawn_ready.append(rank)
                try:
                    sm.rollout_replicas.append(replica)
                    sm.server_handles.append(replica.server_handle)
                    sm.server_addresses.append(replica.server_address)
                except Exception:  # noqa: BLE001
                    pass
                self.log("spawn_ready", **{k: v for k, v in rec.items() if not k.startswith("mono")})
                if self.join_on_ready:
                    self._join_on_ready(rank, replica, rec)
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)[:800]
                self.log("spawn_error", rank=rank, error=rec["error"])
            finally:
                self.spawning = False
                self._report()

        threading.Thread(target=run, name=f"pool-spawn-{rank}", daemon=True).start()

    # ------------------------------------------------------------------ the sync wrapper
    async def around(self, global_steps: Optional[int]) -> dict:
        if not self.join_on_ready:
            return await self._around(global_steps)
        t0 = time.perf_counter()
        with self._sync_lock:
            waited = time.perf_counter() - t0
            if waited > 0.5:
                self.log("sync_waited_for_join", waited_s=waited, global_steps=global_steps)
            try:
                return await self._around(global_steps)
            finally:
                self._actor_updated_pending = False

    async def _around(self, global_steps: Optional[int]) -> dict:
        idx = self.n_sync
        self._wrap_logger_once()
        joiners: list[int] = []
        desired = None
        if not self.initial_done:
            # Sync 0 (on_init_end): park the replicas beyond initial_active BEFORE their first Raiden registration
            # (TPU-M1 J1c path: a parked replica registers inside the sync that first includes it).
            for rank in list(self.active[self.initial_active :]):
                await self.leave(rank, "initial park")
            self.initial_done = True
        else:
            # Pre-sync health sweep: a replica that died since the watchdog's last pass must not be addressed by this
            # sync (the first failure act showed the sync reaching a dead unit 12 s after the kill, before a 20 s-timeout
            # probe could fail twice, and the job died in the transfer). One short no-op RPC per active replica.
            await self._presync_health()
            # Repairs first: a dead replica is already out of the LB; take it out of the membership before the sync.
            for rank in sorted(set(self.dead) - self.dead_in_manager):
                r = self.all[rank]
                self.manager.remove_replicas([r])
                self.dead_in_manager.add(rank)
                if rank in self.active:
                    self.active.remove(rank)
                self.log("dead_removed", rank=rank, active=list(self.active),
                         manager_replicas=[x.replica_rank for x in self.manager.replicas])
                if self.cfg.get("destroy_dead", True):
                    await self.destroy(rank, "dead")
            try:
                desired = int(ray.get(self.ctrl.get_desired.remote(), timeout=30))
            except Exception as e:  # noqa: BLE001
                self.log("desired_error", error=repr(e)[:200])
            if desired is not None:
                n_active = len(self.active)
                if desired > n_active:
                    cand = self.spawn_ready[0] if self.spawn_ready else (self.parked[0] if self.parked else None)
                    if cand is not None:
                        await self.join_before_sync(cand, f"scale_out desired={desired} active={n_active}")
                        joiners.append(cand)
                    elif self.cfg.get("spawn", False) and not self.spawning:
                        if self.next_rank < int(self.cfg.get("max_rank", 8)):
                            self.start_spawn(t_request=self._desired_since(desired))
                        else:
                            self.log("spawn_refused", reason="max_rank reached", next_rank=self.next_rank)
                    else:
                        self.log("hold", reason="no parked/ready replica", desired=desired, active=n_active,
                                 spawning=self.spawning)
                elif desired < n_active and n_active > self.min:
                    victim = self.active[-1]
                    try:
                        release = bool(ray.get(self.ctrl.get_release.remote(), timeout=30))
                    except Exception:  # noqa: BLE001
                        release = False
                    if release:
                        await self.destroy(victim, f"scale_in release desired={desired} active={n_active}")
                    else:
                        await self.leave(victim, f"scale_in desired={desired} active={n_active}")
        units = [x.replica_rank for x in self.manager.replicas]
        sampleable = self._sampleable()
        self.log("sync_begin", global_steps=global_steps, manager_replicas=units, desired=desired,
                 joiners=joiners, sampleable=sampleable, active=list(self.active), parked=list(self.parked))
        if not self.manager.replicas:
            # No live replica (the only one was destroyed after a failure and its replacement is still initialising): a
            # Raiden sync has no destination ("dst_units must not be empty" ended tpum3-r2c at version 13). Skip it; the
            # replacement receives this version through its on-demand joiner push (join_on_ready) or the next sync.
            self.last_sync_wall = 0.0
            self.last_sync_version = int(global_steps or 0)
            self.log("sync_skipped_no_replicas", global_steps=global_steps, spawning=self.spawning,
                     spawn_ready=list(self.spawn_ready))
            self.n_sync += 1
            self._report(sampleable=sampleable, last_sync_phases={})
            return {}
        t0 = time.perf_counter()
        metrics = await update_raiden_weights(self.manager, global_steps=global_steps)
        sync_s = time.perf_counter() - t0
        self.last_sync_wall = sync_s
        self.last_sync_version = int(global_steps or 0)
        for rank in self.active:
            self.versions[rank] = int(global_steps or 0)
        keep = {k.split("/")[-1]: v for k, v in (metrics or {}).items() if k.startswith("timing_s/tpu-sync/")}
        self.log("sync_end", global_steps=global_steps, manager_replicas=units, sync_s=sync_s, phases=keep)
        for rank in joiners:
            rec = await self.join_after_sync(rank, global_steps)
            sp = self.spawns.get(rank)
            if sp is not None and "mono_ready" in sp:
                now = time.perf_counter()
                sp["joined_sync"] = idx
                sp["ready_to_lb_add_s"] = now - sp["mono_ready"]
                sp["req_to_first_token_s"] = (now - sp["t_request"]) if sp.get("t_request") else None
                sp["first_token_s"] = rec.get("first_token_s")
                self.log("spawn_joined", **{k: v for k, v in sp.items() if not k.startswith("mono")})
        self.n_sync += 1
        self._report(sampleable=sampleable, last_sync_phases=keep)
        return metrics

    async def _presync_health(self) -> None:
        timeout = float(self.cfg.get("presync_timeout_s", 5.0))
        for rank in list(self.active):
            if rank in self.dead:
                continue
            r = self.all.get(rank)
            if r is None:
                continue
            t0 = time.perf_counter()
            try:
                await asyncio.wait_for(r.server_handle.collective_rpc.remote(method="check_health"), timeout=timeout)
            except Exception as e:  # noqa: BLE001
                t_detect = time.perf_counter()
                is_timeout = isinstance(e, (asyncio.TimeoutError, TimeoutError)) or "Timeout" in repr(e)
                states = self._actor_states(r) if (self.join_on_ready and is_timeout) else None
                if states is not None and states and all(v == "ALIVE" for v in states.values()):
                    self.log("presync_slow", rank=rank, probe_s=t_detect - t0, error=repr(e)[:200], actor_states=states)
                    continue
                info = {"t_detect": _utc(), "mono_detect": t_detect, "probe_s": t_detect - t0, "error": repr(e)[:400],
                        "source": "presync", "actor_states": states}
                try:
                    await self.lb.remove_servers.remote([r.server_address])
                    info["lb_removed_s"] = time.perf_counter() - t_detect
                except Exception as e2:  # noqa: BLE001
                    info["lb_remove_error"] = repr(e2)[:200]
                with self._lock:
                    self.dead[rank] = info
                self.log("dead_detected", rank=rank, **info, active=list(self.active))

    def _early_spawn_poller(self) -> None:
        poll = float(self.cfg.get("early_spawn_poll_s", 5.0))
        while not self._stop.is_set():
            try:
                if self.initial_done and not self.spawning:
                    desired = int(ray.get(self.ctrl.get_desired.remote(), timeout=10))
                    with self._spawn_gate:
                        with self._lock:
                            have = len(self.active) + len(self.parked) + len(self.spawn_ready)
                        if desired > have and not self.spawning and self.next_rank < int(self.cfg.get("max_rank", 8)):
                            self.log("early_spawn", desired=desired, active=len(self.active), spawn_ready=len(self.spawn_ready))
                            self.start_spawn(t_request=self._desired_since(desired))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[POOL] early spawn poller: {e}")
            time.sleep(poll)

    def _desired_since(self, desired: int) -> Optional[float]:
        """perf_counter of the moment the controller first asked for ``desired`` (for the request->first-token clock)."""
        try:
            st = ray.get(self.ctrl.status.remote(), timeout=10)
            reqs = [q for q in st.get("scale_requests", []) if int(q.get("desired", -1)) == desired]
            if reqs:
                wall = float(reqs[-1]["t"])
                return time.perf_counter() - (time.time() - wall)
        except Exception:  # noqa: BLE001
            return None
        return None


async def update_weights(manager: Any, global_steps: Optional[int] = None) -> dict[str, float]:
    """``update_raiden_weights`` behind the pool hook (``VERL_TPU_POOL_CONTROL``) or the spike (``VERL_TPU_LATE_JOIN_SPIKE``)."""
    raw = os.environ.get(CONTROL_ENV, "").strip()
    if raw:
        hook = getattr(manager, "_pool_hook", None)
        if hook is None:
            hook = PoolHook(manager, json.loads(raw))
            manager._pool_hook = hook
        return await hook.around(global_steps)
    if os.environ.get(SPIKE_ENV, "").strip():
        from verl_hardware_plugin.accelerators.tpu.engines.late_join_spike import update_weights as spike_update

        return await spike_update(manager, global_steps=global_steps)
    return await update_raiden_weights(manager, global_steps=global_steps)


# ---------------------------------------------------------------------- client-side retry on a dead server
_DEAD_MARKERS = ("EngineDead", "ActorDied", "RayActorError", "ActorUnavailable", "Engine core", "EngineCore", "died")


def _looks_dead(e: BaseException) -> bool:
    try:
        if isinstance(e, ray.exceptions.RayActorError | ray.exceptions.ActorUnavailableError):
            return True
    except AttributeError:
        if isinstance(e, ray.exceptions.RayActorError):
            return True
    text = f"{type(e).__name__}: {e!r}"
    return any(m in text for m in _DEAD_MARKERS)


def patch_client_retry_on_dead_server(max_attempts: int = 60, sleep_s: float = 1.0) -> bool:
    """``LLMServerClient.generate`` retries (re-acquiring a server) when the server it hit is dead.

    Stock verl propagates the error into the agent loop (``asyncio.gather`` without return_exceptions). With this
    patch the request is re-routed once the watchdog has removed the dead server from the load balancer (the sticky
    entry is dropped on removal). Applied in every verl process when ``VERL_TPU_POOL_RETRY_DEAD=1``.
    """
    try:
        from verl.workers.rollout import llm_server
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[POOL] client retry patch skipped: {e}")
        return False
    cls = llm_server.LLMServerClient
    if getattr(cls, "_pool_retry_patched", False):
        return True
    orig = cls.generate

    async def generate(self, request_id, **kwargs):
        attempt = 0
        while True:
            try:
                return await orig(self, request_id, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                attempt += 1
                if attempt >= max_attempts or not _looks_dead(e):
                    raise
                logger.warning(f"[POOL] request {request_id}: server error ({type(e).__name__}), retry {attempt} in {sleep_s}s")
                await asyncio.sleep(sleep_s)

    cls.generate = generate
    cls._pool_retry_patched = True
    return True
