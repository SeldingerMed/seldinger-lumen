"""Throughput measurement for the batched fast tier (doc §3.3, §3.10 target).

The bible sets an explicit Layer-0 throughput target: "thousands of parallel
environments ... 1e4 or more aggregate environment-steps per second on a single
workstation-class GPU for a single-device navigation task". That figure is a CUDA
number; this helper measures the aggregate env-steps/s of a ``NewtonGuidewireSim``
on whatever device it is on, so the target is verifiable (on a CUDA box) and
regressions in the batched hot loop are catchable (on any device).

Why this matters as a guard, not just a benchmark: the batched step is ~97%
Newton-VBD-kernel time with no host round-trip — but a single stray ``.numpy()``
(a device→host sync) inside the substep loop would silently serialize every env
and tank throughput. The observable signature is that the per-env-step cost stops
falling as the batch grows; ``test_throughput`` pins exactly that property.
"""

from __future__ import annotations

import time

import warp as wp


def measure_throughput(sim, steps: int = 20, warmup: int = 2, dt: float = 2.5e-2,
                       substeps: int = 3, **action) -> dict:
    """Aggregate env-steps/s for ``sim`` over ``steps`` timed steps (after ``warmup``).

    Actuation passes through as ``**action`` (e.g. ``insertion=1.0``). The device is
    synchronized around the timed region so CUDA's async launches are measured as
    executed work, not just enqueued — without this, GPU numbers are meaningless.

    Returns a dict: ``n_envs``, ``device``, ``env_steps_per_s``, ``ms_per_step``,
    ``us_per_env_step``.
    """
    for _ in range(warmup):                  # absorb Warp JIT compile + first-touch allocs
        sim.step(dt=dt, substeps=substeps, **action)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step(dt=dt, substeps=substeps, **action)
    wp.synchronize()                         # wait for the async GPU work before stopping
    elapsed = time.perf_counter() - t0
    E = sim.n_envs
    return {
        "n_envs": E,
        "device": sim.device,
        "env_steps_per_s": steps * E / elapsed,
        "ms_per_step": 1e3 * elapsed / steps,
        "us_per_env_step": 1e6 * elapsed / steps / E,
    }
