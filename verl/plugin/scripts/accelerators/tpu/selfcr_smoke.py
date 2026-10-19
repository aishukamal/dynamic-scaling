#!/usr/bin/env python3
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Smoke test for in-process libtpu suspend / resume (self checkpoint / restore).

Validates the primitive the TPU colocated mode is built on: a TPU process sending CHECKPOINT and
RESTORE to its OWN libtpu control pipes. Run inside a TPU pod or VM with
``LIBTPU_CHECKPOINTING_ENABLED=true``::

    python selfcr_smoke.py --runtime torch    # torch_tpu style workload
    python selfcr_smoke.py --runtime jax      # JAX style workload

Pass criteria: suspend succeeds (devices released), resume succeeds, and the post-resume computation
matches the pre-suspend value bit-exactly, for every cycle.
"""

import argparse
import sys
import time


def compute_jax():
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4096, 4096), dtype=jnp.float32)

    @jax.jit
    def f(m):
        return jnp.sum(m @ m.T)

    return lambda: float(f(x))


def compute_torch():
    import torch
    import torch_tpu  # noqa: F401

    torch.manual_seed(0)
    x = torch.randn(4096, 4096, dtype=torch.float32).to("tpu")

    def run():
        result = (x @ x.T).sum()
        torch.tpu.synchronize()
        return float(result.cpu())

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=["jax", "torch"], required=True)
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()

    from verl_hardware_plugin.accelerators.tpu import tpu_control
    from verl_hardware_plugin.accelerators.tpu.platform_tpu import resume_device, suspend_device

    run = compute_jax() if args.runtime == "jax" else compute_torch()
    baseline = run()
    print(f"[smoke] baseline value: {baseline!r}")

    channel = tpu_control.find_self_channel()
    if channel is None:
        print("[smoke] FAIL: no libtpu control thread (LIBTPU_CHECKPOINTING_ENABLED not set?)")
        sys.exit(1)
    print(f"[smoke] control pipes: request_write_fd={channel[0]}, response_read_fd={channel[1]}")

    for cycle in range(1, args.cycles + 1):
        info = suspend_device()
        print(
            f"[smoke] cycle {cycle}: suspend OK in {info.get('suspend_s', 0):.2f}s, "
            f"image {info.get('rss_delta_mib', 0):.0f} MiB"
        )
        time.sleep(2)  # pause between the cycles
        info = resume_device()
        value = run()
        match = value == baseline
        print(
            f"[smoke] cycle {cycle}: resume OK in {info.get('resume_s', 0):.2f}s, "
            f"value {value!r} {'==' if match else '!='} baseline -> {'PASS' if match else 'FAIL'}"
        )
        if not match:
            sys.exit(1)

    print(f"[smoke] PASS: {args.cycles} suspend/resume cycles, computation bit-exact")


if __name__ == "__main__":
    main()
