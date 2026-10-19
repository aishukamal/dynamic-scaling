# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""PlatformTPU.all_reduce: MAX / MIN of small integer tensors through a SUM-only collective backend."""

from unittest import mock

import pytest
import torch
import torch.distributed as dist

from verl_hardware_plugin.accelerators.tpu import platform_tpu


class _SumWorld:
    """Simulates ``dist.all_reduce(SUM)`` over ``values`` (one tensor per rank), rank by rank."""

    def __init__(self, values):
        self.values = values  # per-rank input tensors
        self.rows_seen = []

    def run(self, op):
        world = len(self.values)
        outs = [v.clone() for v in self.values]
        # every rank's zero matrix with its own row filled; the SUM is the element-wise sum of all of them
        contributions = []

        def make_sum(rank):
            def sum_all_reduce(rows):
                contributions.append(rows.clone())
                self.rows_seen.append(rank)
                # the SUM every rank sees once all ranks contributed
                full = torch.zeros_like(rows)
                for r in range(world):
                    flat = self.values[r].reshape(-1).to(torch.float32)
                    full[r] = flat
                rows.copy_(full)

            return sum_all_reduce

        for rank in range(world):
            platform_tpu.reduce_int_minmax_via_sum(outs[rank], op, world_size=world, rank=rank, sum_all_reduce=make_sum(rank))
        return outs


@pytest.mark.parametrize("op", [dist.ReduceOp.MAX, dist.ReduceOp.MIN])
def test_int_minmax_via_sum_matches_torch(op):
    values = [
        torch.tensor([3, -7, 4096], dtype=torch.long),
        torch.tensor([9, -2, 2048], dtype=torch.long),
        torch.tensor([1, -9, 8192], dtype=torch.long),
    ]
    stacked = torch.stack(values)
    expected = stacked.max(dim=0).values if op == dist.ReduceOp.MAX else stacked.min(dim=0).values
    outs = _SumWorld(values).run(op)
    for out in outs:
        assert torch.equal(out, expected) and out.dtype == torch.long


def test_int_minmax_via_sum_scalar_flag_and_limits():
    flags = [torch.tensor([1]), torch.tensor([0]), torch.tensor([1])]
    assert all(int(o) == 0 for o in _SumWorld(flags).run(dist.ReduceOp.MIN))  # AND of flags via MIN
    assert all(int(o) == 1 for o in _SumWorld(flags).run(dist.ReduceOp.MAX))
    scalar = [torch.tensor(5), torch.tensor(11)]  # 0-d tensors keep their shape
    outs = _SumWorld(scalar).run(dist.ReduceOp.MAX)
    assert [int(o) for o in outs] == [11, 11] and outs[0].shape == ()
    with pytest.raises(ValueError):
        _SumWorld([torch.tensor([2**24]), torch.tensor([1])]).run(dist.ReduceOp.MAX)


def test_platform_all_reduce_routes_minmax_ints_and_passes_the_rest(monkeypatch):
    calls = []
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None, group=None: calls.append((op, group)))
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    platform = platform_tpu.PlatformTPU.__new__(platform_tpu.PlatformTPU)
    t = torch.tensor([7], dtype=torch.long)
    platform.all_reduce(t, op=dist.ReduceOp.MAX, group="g")
    assert calls == [(dist.ReduceOp.SUM, "g")] and int(t) == 7  # SUM of a one-rank world: unchanged value
    f = torch.tensor([1.5])
    platform.all_reduce(f, op=dist.ReduceOp.MAX, group="g")  # floats go straight through
    s = torch.tensor([2], dtype=torch.long)
    platform.all_reduce(s, op=dist.ReduceOp.SUM, group="g")  # SUM goes straight through
    assert calls[1:] == [(dist.ReduceOp.MAX, "g"), (dist.ReduceOp.SUM, "g")]
