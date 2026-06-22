"""Batched throughput: env-steps/s for the Newton guidewire sim vs env count.

One Newton model holds E independent rods in a shared vessel; one solver.step
advances them all. On a GPU this is the RL-throughput path; on CPU it still shows
the per-step cost amortizing across envs. Pass --device cuda on a GPU box.

    python -m benchmarks.throughput --device cpu --envs 1 8 64
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from lumen.newton.sim import NewtonGuidewireSim


def _scene(n=11, M=40, L=80.0, R=2.0):
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.full(n, 1.6), np.zeros(n), np.linspace(4, 4 + 2 * (n - 1), n)], axis=1)
    return vessel, R, dev


def measure(n_envs, device, steps=50, warmup=5, substeps=5):
    vessel, R, dev = _scene()
    sim = NewtonGuidewireSim(vessel, R, dev, radius=0.2, kappa=3e3, d_hat=0.3,
                             n_envs=n_envs, device=device)
    for _ in range(warmup):                              # JIT compile + caches warm
        sim.step(dt=2.5e-2, substeps=substeps, insertion=0.2)
    import warp as wp
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        sim.step(dt=2.5e-2, substeps=substeps, insertion=0.2)
    wp.synchronize()
    dt = time.perf_counter() - t0
    env_steps = steps * n_envs
    return env_steps / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 8, 64])
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    print(f"device={args.device}  steps={args.steps}")
    print(f"{'n_envs':>8} {'env-steps/s':>14}")
    for E in args.envs:
        rate = measure(E, args.device, steps=args.steps)
        print(f"{E:>8} {rate:>14,.0f}")


if __name__ == "__main__":
    main()
