"""Trainer-side replica membership: one lock for syncs and membership changes, plus versions.

verl's ``CheckpointEngineManager.update_weights`` (verl/checkpoint_engine/base.py) rebuilds the
temporary rollout worker group from ``self.replicas`` on EVERY call. If ``replicas`` changes while
a sync is in flight (between the ``await self.abort_replicas()`` and the ``ray.get`` that drives
the NCCL/NIXL process group), the process group topology and the replica set disagree and the
sync hangs or corrupts. Rule enforced here: membership (attach/detach) and weight syncs (regular or
joiner-only) are mutually exclusive via a single ``asyncio.Lock``.

This module is pure Python so the locking and version-gating semantics can be unit tested without
verl. The manager objects are duck-typed: anything with ``replicas`` (list), ``add_replicas(list)``,
``remove_replicas(list)`` and ``async update_weights(global_steps=...)`` works, which matches
``CheckpointEngineManager``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable


@dataclass
class SyncRecord:
    kind: str  # "regular" | "joiner"
    global_steps: int
    replica_ids: list[str]
    started_at: float
    finished_at: float

    @property
    def wall_s(self) -> float:
        return self.finished_at - self.started_at


@dataclass
class MembershipRegistry:
    """Bookkeeping shared by the trainer subclass and the locked checkpoint manager.

    Attributes:
        lock: the one lock. Held for the whole duration of any sync and of any attach/detach.
        members: ordered replica ids currently attached (mirrors ``manager.replicas``).
        versions: replica id -> param version last synced INTO that replica.
        last_synced_version: param version of the most recent completed sync of any kind.
        history: completed syncs (bounded).
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    members: list[str] = field(default_factory=list)
    versions: dict[str, int] = field(default_factory=dict)
    last_synced_version: int | None = None
    history: list[SyncRecord] = field(default_factory=list)
    max_history: int = 64
    sync_in_progress: bool = False

    def mark_synced(self, replica_ids: Iterable[str], global_steps: int, kind: str, started_at: float) -> SyncRecord:
        ids = list(replica_ids)
        for rid in ids:
            self.versions[rid] = global_steps
        if self.last_synced_version is None or global_steps >= self.last_synced_version:
            self.last_synced_version = global_steps
        rec = SyncRecord(kind=kind, global_steps=global_steps, replica_ids=ids, started_at=started_at, finished_at=time.time())
        self.history.append(rec)
        if len(self.history) > self.max_history:
            del self.history[: len(self.history) - self.max_history]
        return rec

    def is_current(self, replica_id: str) -> bool:
        """Version gate: a replica may enter the load balancer only if it is at least as fresh as
        the newest completed sync. Before any sync has completed everything is 'current'."""
        if self.last_synced_version is None:
            return True
        return self.versions.get(replica_id, -1) >= self.last_synced_version

    def snapshot(self) -> dict[str, Any]:
        last = self.history[-1] if self.history else None
        return {
            "members": list(self.members),
            "versions": dict(self.versions),
            "last_synced_version": self.last_synced_version,
            "sync_in_progress": self.sync_in_progress,
            "last_sync_wall_s": last.wall_s if last else None,
            "last_sync_kind": last.kind if last else None,
        }


KeyFn = Callable[[Any], str]


def default_key(replica: Any) -> str:
    """Stable identity for a RolloutReplica across Ray pickling boundaries: the replica_rank.

    Replica objects are pickled when they cross actor boundaries (rollouter -> trainer), so object
    identity is useless. ``replica_rank`` is unique because the rollouter never reuses ranks.
    """
    rank = getattr(replica, "replica_rank", None)
    if rank is None:
        raise TypeError(f"replica {replica!r} has no replica_rank; pass key_fn explicitly")
    return f"standalone_{rank}"


async def attach(registry: MembershipRegistry, manager: Any, replicas: list[Any], key_fn: KeyFn = default_key) -> list[str]:
    """Add ``replicas`` to the manager's membership under the lock. Idempotent per id."""
    async with registry.lock:
        added: list[str] = []
        new_objs: list[Any] = []
        for r in replicas:
            rid = key_fn(r)
            if rid in registry.members:
                continue
            registry.members.append(rid)
            new_objs.append(r)
            added.append(rid)
        if new_objs:
            manager.add_replicas(new_objs)
        return added


async def detach(registry: MembershipRegistry, manager: Any, replica_ids: list[str], key_fn: KeyFn = default_key) -> list[str]:
    """Remove replicas (by id) from the manager's membership under the lock. Idempotent per id."""
    async with registry.lock:
        wanted = set(replica_ids)
        victims = [r for r in list(manager.replicas) if key_fn(r) in wanted]
        if victims:
            manager.remove_replicas(victims)
        removed = [rid for rid in registry.members if rid in wanted]
        registry.members = [rid for rid in registry.members if rid not in wanted]
        return removed


async def locked_update_weights(
    registry: MembershipRegistry,
    manager: Any,
    global_steps: int,
    key_fn: KeyFn = default_key,
    kind: str = "regular",
    do_sync: Callable[[], Awaitable[Any]] | None = None,
) -> Any:
    """Run a sync under the lock and record versions for every replica that took part.

    ``do_sync`` defaults to ``manager.update_weights(global_steps=...)``; the locked checkpoint
    manager subclass passes ``super().update_weights`` to avoid recursion.
    """
    async with registry.lock:
        registry.sync_in_progress = True
        started = time.time()
        try:
            if do_sync is None:
                result = await manager.update_weights(global_steps=global_steps)
            else:
                result = await do_sync()
        finally:
            registry.sync_in_progress = False
        registry.mark_synced([key_fn(r) for r in manager.replicas], global_steps, kind=kind, started_at=started)
        return result


async def sync_joiners(
    registry: MembershipRegistry,
    make_manager: Callable[[list[Any]], Any],
    manager: Any,
    joiner_ids: list[str],
    global_steps: int,
    key_fn: KeyFn = default_key,
) -> int:
    """Joiner-only sync: build a temporary manager over just the joiners and sync them.

    Joiners must already be attached (so that a regular sync that happens to run first also covers
    them). Joiners that are already at ``global_steps`` or newer are skipped. Returns the version
    the joiners are at after the call.
    """
    async with registry.lock:
        wanted = set(joiner_ids)
        missing = wanted - set(registry.members)
        if missing:
            raise ValueError(f"joiners not attached: {sorted(missing)}")
        stale = [r for r in manager.replicas if key_fn(r) in wanted and registry.versions.get(key_fn(r), -1) < global_steps]
        if not stale:
            return max(registry.versions.get(rid, -1) for rid in wanted)
        registry.sync_in_progress = True
        started = time.time()
        try:
            tmp = make_manager(stale)
            await tmp.update_weights(global_steps=global_steps)
        finally:
            registry.sync_in_progress = False
        registry.mark_synced([key_fn(r) for r in stale], global_steps, kind="joiner", started_at=started)
        return global_steps
