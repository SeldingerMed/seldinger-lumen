"""Batched throughput benchmark for the Layer-0 fast tier (doc §3.3 target).

Sweeps the env count for a single-device navigation task and reports aggregate
env-steps/s on the detected device. The bible's target is >=1e4 aggregate
env-steps/s on a workstation GPU; on CPU (Warp's CPU backend, a handful of
threads) the absolute rate is far lower, but the *scaling* — per-env cost falling
as the batch grows — is the property that carries to the GPU.

    python examples/benchmark_throughput.py            # detected device
    python examples/benchmark_throughput.py --device cuda --envs 1,256,1024,4096
"""

from __future__ import annotations

import argparse

import numpy as np

from lumen.hardware import detect_device
from lumen.newton.sim import NewtonGuidewireSim
from lumen.newton.throughput import measure_throughput

TARGET = 1e4  # bible §3.3: aggregate env-steps/s on a workstation GPU


def _scene(M=30, L=60.0, n=9):
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.zeros(n), np.zeros(n), np.linspace(4, 4 + 2 * (n - 1), n)], axis=1)
    return vessel, dev


def main() -> None:
    def _positive_int(s):
        v = int(s)
        if v <= 0:
            raise argparse.ArgumentTypeError(f"must be a positive integer; got {s}")
        return v

    def _env_list(s):
        envs = [int(x) for x in s.split(",")]
        if not envs or any(e <= 0 for e in envs):
            raise argparse.ArgumentTypeError(f"--envs must be positive counts; got {s!r}")
        return envs

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="cpu | cuda (default: detected)")
    ap.add_argument("--envs", default="1,16,64,256", type=_env_list,
                    help="comma-separated positive env counts")
    ap.add_argument("--steps", type=_positive_int, default=20)
    ap.add_argument("--substeps", type=_positive_int, default=3)
    args = ap.parse_args()

    device = args.device or detect_device()
    vessel, dev = _scene()
    print(f"device={device}  steps={args.steps}  substeps={args.substeps}  "
          f"(target >= {TARGET:.0e} env-steps/s on a workstation GPU)\n")
    print(f"{'n_envs':>8} {'env-steps/s':>14} {'ms/step':>10} {'us/env-step':>14}")
    best = 0.0
    for E in args.envs:
        sim = NewtonGuidewireSim(vessel, 2.0, dev, radius=0.2, n_envs=E, device=device)
        r = measure_throughput(sim, steps=args.steps, substeps=args.substeps, insertion=1.0)
        best = max(best, r["env_steps_per_s"])
        print(f"{E:>8} {r['env_steps_per_s']:>14.0f} {r['ms_per_step']:>10.1f} "
              f"{r['us_per_env_step']:>14.1f}")
    hit = "MET" if best >= TARGET else "below target (expected on CPU; GPU is the target)"
    print(f"\npeak aggregate: {best:.0f} env-steps/s  [{hit}]")


if __name__ == "__main__":
    main()
