"""GPU session: CPU↔CUDA parity + batched throughput. Run on a GPU box.

    python -m benchmarks.gpu_session

Validates that the batched on-device kernels (contact, HGO wall, clot, flow) produce
the same result on CUDA as on the Warp-CPU backend, then sweeps env-steps/s vs env
count for the rigid / deformable / clot+flow stacks. The CPU path is the reference
the test suite checks; this confirms the GPU path matches and measures throughput.
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

from lumen.newton.flow import FlowField, FlowFieldParams
from lumen.newton.sim import NewtonGuidewireSim


def _scene(n=11, M=60, L=120.0, R=2.0):
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.full(n, 1.6), np.zeros(n), np.linspace(4, 4 + 2 * (n - 1), n)], axis=1)
    return vessel, R, dev


def _make(device, E, mode):
    vessel, R, dev = _scene()
    kw = dict(radius=0.2, kappa=3e3, d_hat=0.3, n_envs=E, device=device, vbd_iterations=10)
    if mode == "deformable":
        kw["deformable_wall"] = True
    elif mode == "clotflow":
        kw.update(flow=FlowField(FlowFieldParams()), clot_segment=(55, 70), clot_height=1.6)
    return NewtonGuidewireSim(vessel, R, dev, **kw)


def _drive(sim, steps, substeps=3):
    E = sim.n_envs
    ins = np.linspace(0.2, 0.6, E) if E > 1 else 0.3
    pre = (20.0, 0.0, 0.0) if sim.clot is None else (0.0, 0.0, 0.0)
    for _ in range(steps):
        sim.step(dt=2.5e-2, substeps=substeps, insertion=ins, preload=pre)


def parity(mode, E=8, steps=20):
    """Max abs CPU vs CUDA difference on node positions after identical rollouts."""
    sc = _make("cpu", E, mode); _drive(sc, steps)
    sg = _make("cuda", E, mode); _drive(sg, steps)
    pc, pg = sc.env_positions(), sg.env_positions()
    dpos = float(np.abs(pc - pg).max())
    dwall = abs(sc.wall_max_deflection() - sg.wall_max_deflection())
    print(f"  parity[{mode:10s}] E={E}: max|Δpos|={dpos:.2e}  Δwall_defl={dwall:.2e}  "
          f"cpu_finite={np.isfinite(pc).all()} cuda_finite={np.isfinite(pg).all()}")
    return dpos


def throughput(mode, device, envs, steps=30, substeps=5):
    print(f"  throughput[{mode}] on {device}:")
    print(f"    {'n_envs':>8} {'env-steps/s':>15}")
    for E in envs:
        sim = _make(device, E, mode)
        for _ in range(5):
            _drive(sim, 1, substeps)          # warm JIT + caches
        wp.synchronize()
        t0 = time.perf_counter()
        _drive(sim, steps, substeps)
        wp.synchronize()
        rate = steps * E / (time.perf_counter() - t0)
        print(f"    {E:>8} {rate:>15,.0f}")


def main():
    wp.init()
    has_cuda = any(d.is_cuda for d in wp.get_devices())
    print(f"warp devices: {[str(d) for d in wp.get_devices()]}  cuda={has_cuda}")

    if has_cuda:
        print("\n== CPU<->CUDA parity ==")
        for m in ("rigid", "deformable", "clotflow"):
            parity(m)

    dev = "cuda" if has_cuda else "cpu"
    # NB: the model is built with a Python add_rod loop (O(E) host calls), so very
    # large E is dominated by build time (amortized out of the rate, but slow to set
    # up). Cap at a moot-but-representative count; per-step throughput is the signal.
    envs = [1, 64, 512, 2048] if has_cuda else [1, 8, 64]
    print(f"\n== throughput ({dev}) ==")
    for m in ("rigid", "deformable", "clotflow"):
        throughput(m, dev, envs)


if __name__ == "__main__":
    main()
