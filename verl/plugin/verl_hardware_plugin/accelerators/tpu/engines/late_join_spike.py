# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Late-join spike (experiment driver, out of tree): can a standalone vLLM-TPU replica that is asleep leave and
re-join a running separate_async job's Raiden weight sync and load balancer, without the trainer stalling.

Active only when ``VERL_TPU_LATE_JOIN_SPIKE`` holds a JSON schedule; otherwise ``update_weights`` is exactly
``update_raiden_weights``. Schedule keys (sync indices count the manager's ``update_weights`` calls, 0 = init):
``{"sleepers": [3], "sleep_after_first_sync": false, "events": [{"sync": 3, "join": 3}, {"sync": 6, "leave": 3}],
"log": "/data/tpu-m1/spike.jsonl", "lb_poll_s": 2.0, "probe": true, "prewake": true}``.

leave  = remove the replica's server from the global load balancer -> abort its in-flight requests with the gate
         rejecting -> vLLM TPUWorker.sleep(level=2) through collective_rpc (weights + KV storage freed; the chip stays
         owned) -> manager.remove_replicas (the next sync's destination units exclude it).
join   = TPUWorker.wake_up(weights, kv_cache) (started right after the previous sync when ``prewake``) ->
         manager.add_replicas before the sync (its workers register with Raiden inside the sync if they never did,
         the transfer includes its units, install writes the weights) -> after the sync: add_servers to the load
         balancer, clear the sticky cache, probe the replica (first token, greedy tokens vs the other replicas,
         receive-buffer norms vs the other replicas).
Everything is appended as JSON lines to ``log`` with UTC timestamps.
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

from verl_hardware_plugin.accelerators.tpu.engines.raiden_checkpoint_engine import update_raiden_weights

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

SCHEDULE_ENV = "VERL_TPU_LATE_JOIN_SPIKE"
# Token ids of a short fixed prompt (Qwen tokenizer); only used to compare replicas with each other.
PROBE_PROMPT = [785, 6722, 315, 9625, 374]
PROBE_TOKENS = 16


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _find_trainer(manager: Any) -> Any:
    """The trainer instance holding ``manager`` as ``standalone_checkpoint_manager`` (no back-reference exists)."""
    for d in gc.get_referrers(manager):
        if isinstance(d, dict) and d.get("standalone_checkpoint_manager") is manager:
            for obj in gc.get_referrers(d):
                if getattr(obj, "__dict__", None) is d and hasattr(obj, "standalone_server_manager"):
                    return obj
    raise RuntimeError("late-join spike: no trainer holds this manager as standalone_checkpoint_manager")


class LateJoinSpike:
    def __init__(self, manager: Any, cfg: dict):
        self.manager = manager
        self.cfg = cfg
        self.log_path = cfg.get("log", "/data/tpu-m1/spike.jsonl")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.all = list(manager.replicas)
        self.by_rank = {r.replica_rank: r for r in self.all}
        self.addr_to_rank = {r.server_address: r.replica_rank for r in self.all}
        trainer = _find_trainer(manager)
        self.lb = trainer.standalone_server_manager.global_load_balancer
        self.events: dict[int, list[dict]] = {}
        for ev in cfg.get("events", []):
            self.events.setdefault(int(ev["sync"]), []).append(ev)
        self.sleepers = [int(x) for x in cfg.get("sleepers", [])]
        self.pending_wake: dict[int, tuple[threading.Thread, dict]] = {}
        self.n_sync = 0
        self.initial_done = False
        self.log("init", replicas=[r.replica_rank for r in self.all], addresses=self.addr_to_rank, cfg=cfg)
        self._stop = threading.Event()
        threading.Thread(target=self._lb_sampler, name="spike-lb-sampler", daemon=True).start()

    # ------------------------------------------------------------------ logging
    def log(self, kind: str, **kw: Any) -> None:
        rec = {"t": _utc(), "mono": time.perf_counter(), "kind": kind, **kw}
        line = json.dumps(rec, default=str)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        logger.info(f"[LATE-JOIN SPIKE] {line}")

    def _lb_sampler(self) -> None:
        poll = float(self.cfg.get("lb_poll_s", 2.0))
        while not self._stop.is_set():
            try:
                st = ray.get(self.lb.get_status.remote(), timeout=10)
                servers = {str(self.addr_to_rank.get(a, a)): n for a, n in st.get("servers", {}).items()}
                rec = {"t": _utc(), "mono": time.perf_counter(), "kind": "lb", "inflight": servers}
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[LATE-JOIN SPIKE] lb sampler: {e}")
            time.sleep(poll)

    # ------------------------------------------------------------------ primitives
    async def leave(self, rank: int, why: str) -> None:
        r = self.by_rank[rank]
        t0 = time.perf_counter()
        await self.lb.remove_servers.remote([r.server_address])
        t1 = time.perf_counter()
        aborted = await r.server_handle.abort_all_requests.remote(reject_request=True)
        t2 = time.perf_counter()
        slept = await r.server_handle.collective_rpc.remote(method="sleep", kwargs={"level": 2})
        t3 = time.perf_counter()
        self.manager.remove_replicas([r])
        self.log(
            "leave", sync=self.n_sync, rank=rank, why=why, lb_remove_s=t1 - t0, abort_s=t2 - t1, sleep_s=t3 - t2,
            total_s=t3 - t0, aborted=aborted, sleep_info=slept, manager_replicas=[x.replica_rank for x in self.manager.replicas],
        )

    def start_wake(self, rank: int) -> None:
        if rank in self.pending_wake:
            return
        r = self.by_rank[rank]
        info: dict[str, Any] = {"t_start": _utc(), "mono_start": time.perf_counter()}

        def run() -> None:
            try:
                info["result"] = ray.get(
                    r.server_handle.collective_rpc.remote(method="wake_up", kwargs={"tags": ["weights", "kv_cache"]})
                )
            except Exception as e:  # noqa: BLE001
                info["error"] = repr(e)
            info["mono_end"] = time.perf_counter()
            info["t_end"] = _utc()

        th = threading.Thread(target=run, name=f"spike-wake-{rank}", daemon=True)
        th.start()
        self.pending_wake[rank] = (th, info)
        self.log("wake_start", sync=self.n_sync, rank=rank, t_start=info["t_start"])

    async def join_before_sync(self, rank: int) -> None:
        r = self.by_rank[rank]
        t0 = time.perf_counter()
        if rank not in self.pending_wake:
            self.start_wake(rank)
        th, info = self.pending_wake.pop(rank)
        await asyncio.to_thread(th.join)
        t1 = time.perf_counter()
        self.manager.add_replicas([r])
        self.log(
            "join_before_sync", sync=self.n_sync, rank=rank, waited_for_wake_s=t1 - t0,
            wake_s=info.get("mono_end", t1) - info["mono_start"], wake=info,
            manager_replicas=[x.replica_rank for x in self.manager.replicas],
        )

    async def _probe(self, r: Any, request_id: str, max_tokens: int) -> tuple[float, list[int]]:
        t0 = time.perf_counter()
        out = await r.server_handle.generate.remote(
            prompt_ids=list(PROBE_PROMPT),
            # logprobs=True as every verl rollout request: without it vllm-torchtpu returns an empty logprobs
            # list and the plugin's _execute_dag (detach_zero_copy_from_model_runner_output) raises
            # "not enough values to unpack (expected 4, got 0)" and kills the engine (seen on the J1a joiner).
            sampling_params={"temperature": 0.0, "max_tokens": max_tokens, "logprobs": True},
            request_id=request_id,
        )
        return time.perf_counter() - t0, list(getattr(out, "token_ids", []) or [])

    async def join_after_sync(self, rank: int) -> None:
        r = self.by_rank[rank]
        t0 = time.perf_counter()
        await self.lb.add_servers.remote({r.server_address: r.server_handle})
        cleared = await self.lb.clear_sticky_cache.remote()
        t1 = time.perf_counter()
        rec: dict[str, Any] = {"lb_add_s": t1 - t0, "cleared": cleared}
        if self.cfg.get("probe", True):
            ft_s, ft_ids = await self._probe(r, f"spike-ft-{rank}-{self.n_sync}", 1)
            rec.update(first_token_s=ft_s, first_token_ids=ft_ids)
            greedy = {}
            for x in self.manager.replicas:
                s, ids = await self._probe(x, f"spike-greedy-{x.replica_rank}-{self.n_sync}", PROBE_TOKENS)
                greedy[x.replica_rank] = {"s": s, "ids": ids}
            others = [v["ids"] for k, v in greedy.items() if k != rank]
            rec.update(greedy=greedy, greedy_equal_to_others=[v == greedy[rank]["ids"] for v in others])
        if self.cfg.get("fingerprint", False):
            # Receive-buffer norms per replica (compiles the norm kernels on first use, about 20 s per replica).
            stats = await asyncio.gather(
                *[x.server_handle.collective_rpc.remote(method="get_model_weights_stats") for x in self.manager.replicas]
            )
            norms = {}
            for x, st in zip(self.manager.replicas, stats, strict=True):
                workers = st if isinstance(st, list | tuple) else [st]
                norms[x.replica_rank] = [
                    {k: w.get(k) for k in ("rank", "total_numel", "l1_norm", "l2_norm")} for w in workers if isinstance(w, dict)
                ]
            rec["receive_buffer_norms"] = norms
        self.log("join_after_sync", sync=self.n_sync, rank=rank, **rec)

    # ------------------------------------------------------------------ the sync wrapper
    async def around(self, global_steps: Optional[int]) -> dict:
        idx = self.n_sync
        evs = self.events.get(idx, [])
        if not self.initial_done and not self.cfg.get("sleep_after_first_sync", False):
            for rank in self.sleepers:
                await self.leave(rank, "initial (before first registration)")
            self.initial_done = True
        for ev in evs:
            if "leave" in ev:
                await self.leave(int(ev["leave"]), f"event sync {idx}")
        for ev in evs:
            if "join" in ev:
                await self.join_before_sync(int(ev["join"]))
        units = [x.replica_rank for x in self.manager.replicas]
        self.log("sync_begin", sync=idx, global_steps=global_steps, manager_replicas=units)
        t0 = time.perf_counter()
        metrics = await update_raiden_weights(self.manager, global_steps=global_steps)
        sync_s = time.perf_counter() - t0
        keep = {k.split("/")[-1]: v for k, v in (metrics or {}).items() if k.startswith("timing_s/tpu-sync/")}
        self.log("sync_end", sync=idx, global_steps=global_steps, manager_replicas=units, sync_s=sync_s, phases=keep)
        if not self.initial_done:
            for rank in self.sleepers:
                await self.leave(rank, "initial (after first registration)")
            self.initial_done = True
        for ev in evs:
            if "join" in ev:
                await self.join_after_sync(int(ev["join"]))
        if self.cfg.get("prewake", True):
            for ev in self.events.get(idx + 1, []):
                if "join" in ev:
                    self.start_wake(int(ev["join"]))
        self.n_sync += 1
        return metrics


async def update_weights(manager: Any, global_steps: Optional[int] = None) -> dict[str, float]:
    """``update_raiden_weights`` with the late-join schedule around it when ``VERL_TPU_LATE_JOIN_SPIKE`` is set."""
    raw = os.environ.get(SCHEDULE_ENV, "").strip()
    if not raw:
        return await update_raiden_weights(manager, global_steps=global_steps)
    spike = getattr(manager, "_late_join_spike", None)
    if spike is None:
        spike = LateJoinSpike(manager, json.loads(raw))
        manager._late_join_spike = spike
    return await spike.around(global_steps)
