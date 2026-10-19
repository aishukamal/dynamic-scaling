"""ScaleOrchestrator: converge the ACTIVE standalone replica count to a desired value.

Pure Python. All side effects go through a ``PoolBackend`` (duck-typed; see ``RayPoolBackend`` in
``control_actor.py`` for the real one and ``tests/test_orchestrator.py`` for the fake).

Ordering rules enforced (from STATE.md):

scale-out, one replica at a time
    1. pick a parked (sleeping) replica if any, else create a new standalone replica
       (``RolloutReplica.init_standalone``), but never beyond ``max_standalone``
    2. wake it (sleep tier; no-op for vLLM standalone at the pinned verl commit)
    3. attach it to the trainer's checkpoint membership (under the membership lock)
    4. joiner-only weight sync (under the same lock) so it has the current param version. The
       trainer may DEFER it (joiner_sync_mode=clean_only and the actor weights have moved since
       the last sync); the attached joiner is then covered by the next regular sync instead.
    5. version gate: the joiner must be at least as fresh as the newest completed sync; the
       orchestrator polls (re-requesting the joiner sync each time) until the gate passes or
       ``join_timeout_s`` elapses, in which case it rolls back (detach + park)
    6. resume generation, register in the load balancer, recompute max_concurrent_samples

scale-in, one replica at a time
    1. refuse to go below ``min_standalone``
    2. remove from the load balancer (cuts routing; partial-rollout retries re-route)
    3. abort in-flight requests on it (engine stays paused)
    4. detach from the checkpoint membership (under the lock)
    5. sleep it (park, warm) or destroy it (release=True: kill actors + placement group)

The orchestrator never changes membership itself; attach/detach/sync are backend calls that the
trainer serialises with its own lock. ``step()`` performs exactly one replica move so a reconciler
thread can interleave status reporting and so that a new desired value is picked up between moves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class PoolBackend(Protocol):
    def list_active(self) -> list[str]: ...
    def list_parked(self) -> list[str]: ...
    def create_standalone(self, n: int) -> list[str]: ...
    def wake(self, rids: list[str]) -> None: ...
    def resume_generation(self, rids: list[str]) -> None: ...
    def attach(self, rids: list[str]) -> None: ...
    def sync_joiners(self, rids: list[str]) -> dict: ...  # {"version": int|None, "deferred": bool, "reason": str}
    def replica_versions(self) -> dict[str, int]: ...
    def last_synced_version(self) -> int | None: ...
    def lb_add(self, rids: list[str]) -> None: ...
    def lb_remove(self, rids: list[str]) -> None: ...
    def abort(self, rids: list[str]) -> None: ...
    def detach(self, rids: list[str]) -> None: ...
    def sleep(self, rids: list[str]) -> None: ...
    def destroy(self, rids: list[str]) -> None: ...
    def evict(self, rids: list[str]) -> None: ...  # LB + list removal without touching dead actors


class ScaleError(RuntimeError):
    pass


@dataclass
class PoolLimits:
    min_standalone: int
    max_standalone: int

    def __post_init__(self) -> None:
        if self.min_standalone < 1:
            # FullyAsyncLLMServerManager.get_client() decides only_hybrid from len(rollout_replicas)
            # at creation time; an empty standalone pool would make every generate() raise.
            raise ValueError("min_standalone must be >= 1")
        if self.max_standalone < self.min_standalone:
            raise ValueError("max_standalone must be >= min_standalone")

    def clamp(self, n: int) -> int:
        return max(self.min_standalone, min(self.max_standalone, int(n)))


@dataclass
class ScaleEvent:
    t: float
    action: str
    replica_id: str | None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"t": self.t, "action": self.action, "replica_id": self.replica_id, **self.detail}


class ScaleOrchestrator:
    def __init__(
        self,
        backend: PoolBackend,
        limits: PoolLimits,
        release_on_scale_in: bool = False,
        join_timeout_s: float = 3600.0,
        join_poll_s: float = 5.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] | None = print,
        max_events: int = 1000,
    ) -> None:
        self.backend = backend
        self.limits = limits
        self.release_on_scale_in = release_on_scale_in
        self.join_timeout_s = join_timeout_s
        self.join_poll_s = join_poll_s
        self.clock = clock
        self.sleep = sleep
        self.log = log or (lambda s: None)
        self.desired: int | None = None
        self.events: list[ScaleEvent] = []
        self.max_events = max_events
        self.last_error: str | None = None
        self.in_progress: bool = False
        self.ops_total = {"scale_out": 0, "scale_in": 0, "rollback": 0, "noop": 0, "evict": 0}

    # ------------------------------------------------------------------ public API
    def active_count(self) -> int:
        return len(self.backend.list_active())

    def parked_count(self) -> int:
        return len(self.backend.list_parked())

    def request(self, standalone: int, release: bool | None = None) -> dict[str, Any]:
        """Record a new desired count (clamped). Idempotent: same value twice is a no-op."""
        if release is not None:
            self.release_on_scale_in = bool(release)
        requested = int(standalone)
        desired = self.limits.clamp(requested)
        active = self.active_count()
        changed = desired != self.desired
        self.desired = desired
        noop = (not changed) or desired == active
        if noop and desired == active:
            self.ops_total["noop"] += 1
        self._event("request", None, requested=requested, desired=desired, active=active, clamped=desired != requested, noop=noop)
        return {"requested": requested, "desired": desired, "active": active, "clamped": desired != requested, "noop": noop}

    def converged(self) -> bool:
        return self.desired is None or self.desired == self.active_count()

    def step(self) -> str | None:
        """Perform at most ONE replica move toward ``desired``. Returns the action or None."""
        if self.desired is None:
            return None
        active = self.active_count()
        if active == self.desired:
            return None
        self.in_progress = True
        try:
            if active < self.desired:
                return self.scale_out_one()
            return self.scale_in_one()
        except Exception as e:  # noqa: BLE001 - recorded, reconciler decides on retry
            self.last_error = f"{type(e).__name__}: {e}"
            self._event("error", None, error=self.last_error)
            raise
        finally:
            self.in_progress = False

    def converge(self, max_steps: int | None = None) -> list[str]:
        actions: list[str] = []
        while True:
            if max_steps is not None and len(actions) >= max_steps:
                break
            a = self.step()
            if a is None:
                break
            actions.append(a)
        return actions

    # ------------------------------------------------------------------ one move each
    def scale_out_one(self) -> str:
        active = self.backend.list_active()
        parked = self.backend.list_parked()
        if len(active) >= self.limits.max_standalone:
            raise ScaleError(f"at max_standalone={self.limits.max_standalone}")
        t0 = self.clock()
        if parked:
            rid = parked[0]
            source = "wake"
        else:
            if len(active) + len(parked) >= self.limits.max_standalone:
                raise ScaleError("no parked replica and total would exceed max_standalone")
            created = self.backend.create_standalone(1)
            if not created:
                raise ScaleError("create_standalone returned nothing")
            rid = created[0]
            source = "create"
        t_created = self.clock()

        # wake -> attach -> joiner sync (or deferred to the next regular sync) -> gate -> LB add
        self.backend.wake([rid])
        self.backend.attach([rid])
        gated = False
        deadline = self.clock() + self.join_timeout_s
        polls = 0
        while True:
            res = self.backend.sync_joiners([rid]) or {}
            v = self.backend.replica_versions().get(rid, -1)
            last = self.backend.last_synced_version()
            if last is None or v >= last:
                gated = True
                break
            polls += 1
            if polls == 1 or polls % 12 == 0:
                self._event("gate_wait", rid, polls=polls, replica_version=v, last_synced_version=last, deferred=bool(res.get("deferred")), reason=res.get("reason"))
            if self.clock() >= deadline:
                break
            self.sleep(self.join_poll_s)
        if not gated:
            # roll back: never register a stale replica
            self.backend.detach([rid])
            self.backend.sleep([rid])
            self.ops_total["rollback"] += 1
            self._event("rollback", rid, reason="version gate not passed before join_timeout_s", polls=polls)
            raise ScaleError(f"version gate not passed for {rid} within {self.join_timeout_s}s")
        t_synced = self.clock()
        self.backend.resume_generation([rid])
        self.backend.lb_add([rid])
        self.ops_total["scale_out"] += 1
        self._event(
            "scale_out",
            rid,
            source=source,
            create_s=t_created - t0,
            join_s=t_synced - t_created,
            total_s=self.clock() - t0,
            gate_polls=polls,
            active=len(self.backend.list_active()),
            version=self.backend.replica_versions().get(rid),
        )
        return "scale_out"

    def scale_in_one(self) -> str:
        active = self.backend.list_active()
        if len(active) <= self.limits.min_standalone:
            raise ScaleError(f"at min_standalone={self.limits.min_standalone}")
        rid = active[-1]  # LIFO: newest joiner leaves first (deterministic, keeps prestart set stable)
        t0 = self.clock()
        self.backend.lb_remove([rid])
        self.backend.abort([rid])
        self.backend.detach([rid])
        if self.release_on_scale_in:
            self.backend.destroy([rid])
            tier = "destroy"
        else:
            self.backend.sleep([rid])
            tier = "sleep"
        self.ops_total["scale_in"] += 1
        self._event("scale_in", rid, tier=tier, total_s=self.clock() - t0, active=len(self.backend.list_active()))
        return "scale_in"

    def evict_dead(self, rid: str, detail: dict[str, Any] | None = None) -> str:
        """Failure path (replica process died): cut routing, detach from the checkpoint membership
        BEFORE the next weight sync, destroy the leftovers. ``desired`` is unchanged, so the next
        ``step()`` replaces it (parked replica if any, else a new one)."""
        t0 = self.clock()
        errs: dict[str, str] = {}
        for name, fn in (("evict", lambda: self.backend.evict([rid])), ("detach", lambda: self.backend.detach([rid])), ("destroy", lambda: self.backend.destroy([rid]))):
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - keep going; the point is to get it out of membership
                errs[name] = f"{type(e).__name__}: {e}"
        self.ops_total["evict"] += 1
        self._event("evict", rid, total_s=self.clock() - t0, errors=errs, active=len(self.backend.list_active()), desired=self.desired, **(detail or {}))
        return "evict"

    # ------------------------------------------------------------------ helpers
    def _event(self, action: str, rid: str | None, **detail: Any) -> None:
        ev = ScaleEvent(t=self.clock(), action=action, replica_id=rid, detail=detail)
        self.events.append(ev)
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]
        self.log(f"[rl_pool] {action} {rid or ''} {detail}")

    def status(self) -> dict[str, Any]:
        return {
            "desired": self.desired,
            "active": self.active_count(),
            "parked": self.parked_count(),
            "min": self.limits.min_standalone,
            "max": self.limits.max_standalone,
            "in_progress": self.in_progress,
            "release_on_scale_in": self.release_on_scale_in,
            "last_error": self.last_error,
            "ops_total": dict(self.ops_total),
            "recent_events": [e.as_dict() for e in self.events[-20:]],
        }
