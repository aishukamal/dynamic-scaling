"""Test doubles: a scripted PoolBackend and a duck-typed checkpoint manager. No verl, no ray."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeReplica:
    replica_rank: int

    def __hash__(self) -> int:
        return hash(self.replica_rank)


class FakeManager:
    """Duck-typed CheckpointEngineManager: replicas list + add/remove + async update_weights."""

    def __init__(self, replicas: list[FakeReplica], sync_delay: float = 0.0) -> None:
        self.replicas = list(replicas)
        self.sync_delay = sync_delay
        self.calls: list[tuple[str, Any]] = []
        self.sync_seen_sizes: list[int] = []
        self.in_sync = False
        self.membership_changed_during_sync = False

    def add_replicas(self, replicas):
        if self.in_sync:
            self.membership_changed_during_sync = True
        self.replicas.extend(replicas)
        self.calls.append(("add", [r.replica_rank for r in replicas]))

    def remove_replicas(self, replicas):
        if self.in_sync:
            self.membership_changed_during_sync = True
        s = set(replicas)
        self.replicas = [r for r in self.replicas if r not in s]
        self.calls.append(("remove", [r.replica_rank for r in replicas]))

    async def update_weights(self, global_steps=None):
        self.in_sync = True
        self.sync_seen_sizes.append(len(self.replicas))
        self.calls.append(("sync", global_steps, [r.replica_rank for r in self.replicas]))
        await asyncio.sleep(self.sync_delay)
        self.in_sync = False
        return {}


@dataclass
class FakeBackend:
    """Scripted PoolBackend. Records every call in order; models versions like the trainer does."""

    active: list[str] = field(default_factory=list)
    parked: list[str] = field(default_factory=list)
    attached: list[str] = field(default_factory=list)
    versions: dict[str, int] = field(default_factory=dict)
    last_synced: int | None = 0
    trainer_version: int = 0
    defer_joiner_sync: bool = False  # emulate joiner_sync_mode=clean_only with dirty weights
    next_rank: int = 100
    calls: list[tuple[str, list[str]]] = field(default_factory=list)
    destroyed: list[str] = field(default_factory=list)
    fail_on: str | None = None

    def _rec(self, op: str, rids: list[str]) -> None:
        self.calls.append((op, list(rids)))
        if self.fail_on == op:
            raise RuntimeError(f"injected failure in {op}")

    def list_active(self):
        return list(self.active)

    def list_parked(self):
        return list(self.parked)

    def create_standalone(self, n):
        ids = []
        for _ in range(n):
            rid = f"standalone_{self.next_rank}"
            self.next_rank += 1
            self.parked.append(rid)
            ids.append(rid)
        self._rec("create_standalone", ids)
        return ids

    def wake(self, rids):
        self._rec("wake", rids)

    def resume_generation(self, rids):
        self._rec("resume_generation", rids)

    def attach(self, rids):
        self._rec("attach", rids)
        for r in rids:
            if r not in self.attached:
                self.attached.append(r)

    def sync_joiners(self, rids):
        self._rec("sync_joiners", rids)
        if self.defer_joiner_sync:
            return {"version": None, "deferred": True, "reason": "dirty"}
        for r in rids:
            assert r in self.attached, "joiner sync before attach"
            self.versions[r] = self.trainer_version
        return {"version": self.trainer_version, "deferred": False, "reason": "synced"}

    def regular_sync(self):
        """Test helper: what the trainer's regular sync does to attached members."""
        self.trainer_version += 1
        for r in self.attached:
            self.versions[r] = self.trainer_version
        self.last_synced = self.trainer_version

    def replica_versions(self):
        return dict(self.versions)

    def last_synced_version(self):
        return self.last_synced

    def lb_add(self, rids):
        self._rec("lb_add", rids)
        for r in rids:
            assert r in self.attached, "LB add before attach"
            assert self.versions.get(r, -1) >= (self.last_synced or 0), "LB add of stale replica"
            self.parked.remove(r)
            self.active.append(r)

    def lb_remove(self, rids):
        self._rec("lb_remove", rids)
        for r in rids:
            self.active.remove(r)
            self.parked.append(r)

    def abort(self, rids):
        self._rec("abort", rids)
        for r in rids:
            assert r not in self.active, "abort while still routable"

    def detach(self, rids):
        self._rec("detach", rids)
        for r in rids:
            assert r not in self.active, "detach while still in LB"
            if r in self.attached:
                self.attached.remove(r)

    def sleep(self, rids):
        self._rec("sleep", rids)
        for r in rids:
            assert r not in self.attached, "sleep while still attached"

    def destroy(self, rids):
        self._rec("destroy", rids)
        for r in rids:
            assert r not in self.attached and r not in self.active
            self.parked.remove(r)
            self.destroyed.append(r)

    def evict(self, rids):
        """Failure path: active -> parked without touching the (dead) actors."""
        self._rec("evict", rids)
        for r in rids:
            if r in self.active:
                self.active.remove(r)
                self.parked.append(r)

    def ops_for(self, rid: str) -> list[str]:
        return [op for op, rids in self.calls if rid in rids]


def prestarted(n_total: int = 4, n_active: int = 1) -> FakeBackend:
    ids = [f"standalone_{i}" for i in range(n_total)]
    b = FakeBackend(active=ids[:n_active], parked=ids[n_active:], attached=ids[:n_active])
    for r in b.attached:
        b.versions[r] = 0
    return b
