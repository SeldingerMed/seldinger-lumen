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
import json
import sys

import numpy as np

from lumen.hardware import detect_device
from lumen.newton.sim import NewtonGuidewireSim
from lumen.newton.throughput import measure_throughput

TARGET = 1e4  # bible §3.3: aggregate env-steps/s on a workstation GPU


def _scene(M=30, L=60.0, n=9):
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.zeros(n), np.zeros(n), np.linspace(4, 4 + 2 * (n - 1), n)], axis=1)
    return vessel, dev


def main(argv=None) -> int:
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
    ap.add_argument("--min-env-steps-per-s", type=float, default=None,
                    help="Fail if peak aggregate throughput is below this value.")
    ap.add_argument("--require-cuda", action="store_true",
                    help="Fail unless the benchmark actually runs on a CUDA device.")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable summary JSON instead of the human table.")
    args = ap.parse_args(argv)

    device = args.device or detect_device()
    vessel, dev = _scene()
    if args.require_cuda and device != "cuda":
        if args.json:
            print(json.dumps({"device": device, "error": "CUDA device required"}, sort_keys=True))
        else:
            print(f"CUDA device required, but selected device is {device!r}", file=sys.stderr)
        return 2
    if not args.json:
        print(f"device={device}  steps={args.steps}  substeps={args.substeps}  "
              f"(target >= {TARGET:.0e} env-steps/s on a workstation GPU)\n")
        print(f"{'n_envs':>8} {'env-steps/s':>14} {'ms/step':>10} {'us/env-step':>14}")
    best = 0.0
    rows = []
    for E in args.envs:
        sim = NewtonGuidewireSim(vessel, 2.0, dev, radius=0.2, n_envs=E, device=device)
        r = measure_throughput(sim, steps=args.steps, substeps=args.substeps, insertion=1.0)
        best = max(best, r["env_steps_per_s"])
        rows.append(r)
        if not args.json:
            print(f"{E:>8} {r['env_steps_per_s']:>14.0f} {r['ms_per_step']:>10.1f} "
                  f"{r['us_per_env_step']:>14.1f}")
    threshold = args.min_env_steps_per_s
    ok = threshold is None or best >= threshold
    hit = "MET" if best >= TARGET else "below target (expected on CPU; GPU is the target)"
    result = {
        "device": device,
        "steps": args.steps,
        "substeps": args.substeps,
        "envs": args.envs,
        "rows": rows,
        "peak_env_steps_per_s": best,
        "target_env_steps_per_s": TARGET,
        "min_env_steps_per_s": threshold,
        "passed": ok,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"\npeak aggregate: {best:.0f} env-steps/s  [{hit}]")
        if threshold is not None:
            status = "PASS" if ok else "FAIL"
            print(f"minimum threshold: {threshold:.0f} env-steps/s  [{status}]")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
