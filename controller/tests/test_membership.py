import asyncio
import unittest

from fakes import FakeManager, FakeReplica

from rl_pool_autoscaler.membership import MembershipRegistry, attach, detach, locked_update_weights, sync_joiners


def run(coro):
    return asyncio.run(coro)


class MembershipLockTests(unittest.TestCase):
    def test_attach_waits_for_in_flight_sync(self):
        async def scenario():
            reg = MembershipRegistry()
            mgr = FakeManager([FakeReplica(0)], sync_delay=0.05)
            reg.members = ["standalone_0"]
            sync_task = asyncio.create_task(locked_update_weights(reg, mgr, 3))
            await asyncio.sleep(0.01)  # sync is in flight and holds the lock
            self.assertTrue(reg.sync_in_progress)
            t0 = asyncio.get_event_loop().time()
            added = await attach(reg, mgr, [FakeReplica(1)])
            waited = asyncio.get_event_loop().time() - t0
            await sync_task
            return mgr, reg, added, waited

        mgr, reg, added, waited = run(scenario())
        self.assertFalse(mgr.membership_changed_during_sync)
        self.assertEqual(added, ["standalone_1"])
        self.assertGreaterEqual(waited, 0.03)
        self.assertEqual(mgr.sync_seen_sizes, [1])  # sync saw the pre-attach membership
        self.assertEqual(reg.versions, {"standalone_0": 3})
        self.assertEqual(reg.last_synced_version, 3)
        self.assertFalse(reg.is_current("standalone_1"))  # joiner not yet synced
        self.assertTrue(reg.is_current("standalone_0"))

    def test_sync_waits_for_in_flight_detach_and_sees_new_membership(self):
        async def scenario():
            reg = MembershipRegistry()
            reps = [FakeReplica(0), FakeReplica(1)]
            mgr = FakeManager(reps)
            reg.members = ["standalone_0", "standalone_1"]
            async with reg.lock:  # emulate a long membership change
                t = asyncio.create_task(locked_update_weights(reg, mgr, 1))
                await asyncio.sleep(0.01)
                self.assertFalse(mgr.sync_seen_sizes)
                mgr.remove_replicas([reps[1]])
                reg.members.remove("standalone_1")
            await t
            return mgr, reg

        mgr, reg = run(scenario())
        self.assertEqual(mgr.sync_seen_sizes, [1])
        self.assertEqual(reg.versions, {"standalone_0": 1})

    def test_attach_detach_idempotent(self):
        async def scenario():
            reg = MembershipRegistry()
            mgr = FakeManager([])
            r = FakeReplica(7)
            a1 = await attach(reg, mgr, [r])
            a2 = await attach(reg, mgr, [r])
            d1 = await detach(reg, mgr, ["standalone_7"])
            d2 = await detach(reg, mgr, ["standalone_7"])
            return reg, mgr, a1, a2, d1, d2

        reg, mgr, a1, a2, d1, d2 = run(scenario())
        self.assertEqual((a1, a2, d1, d2), (["standalone_7"], [], ["standalone_7"], []))
        self.assertEqual(reg.members, [])
        self.assertEqual(mgr.replicas, [])

    def test_joiner_sync_only_touches_stale_joiners(self):
        async def scenario():
            reg = MembershipRegistry()
            base = [FakeReplica(0)]
            mgr = FakeManager(base)
            reg.members = ["standalone_0"]
            await locked_update_weights(reg, mgr, 5)
            j = FakeReplica(1)
            await attach(reg, mgr, [j])
            made = []

            def make(reps):
                m = FakeManager(reps)
                made.append(m)
                return m

            v = await sync_joiners(reg, make, mgr, ["standalone_1"], 5)
            v2 = await sync_joiners(reg, make, mgr, ["standalone_1"], 5)  # already current: no new manager
            return reg, made, v, v2

        reg, made, v, v2 = run(scenario())
        self.assertEqual((v, v2), (5, 5))
        self.assertEqual(len(made), 1)
        self.assertEqual([r.replica_rank for r in made[0].replicas], [1])  # temp manager over joiners only
        self.assertTrue(reg.is_current("standalone_1"))
        self.assertEqual(reg.history[-1].kind, "joiner")

    def test_joiner_sync_requires_attach_first(self):
        async def scenario():
            reg = MembershipRegistry()
            mgr = FakeManager([])
            with self.assertRaises(ValueError):
                await sync_joiners(reg, FakeManager, mgr, ["standalone_9"], 1)

        run(scenario())

    def test_regular_sync_after_attach_makes_joiner_current(self):
        async def scenario():
            reg = MembershipRegistry()
            mgr = FakeManager([FakeReplica(0)])
            reg.members = ["standalone_0"]
            await attach(reg, mgr, [FakeReplica(1)])
            await locked_update_weights(reg, mgr, 2)
            return reg

        reg = run(scenario())
        self.assertEqual(reg.versions, {"standalone_0": 2, "standalone_1": 2})
        self.assertTrue(reg.is_current("standalone_1"))


if __name__ == "__main__":
    unittest.main()
