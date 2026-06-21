"""Rod-primitive bake-off harness (doc §3.4.1 / M4).

The doc calls for a focused bake-off of device-rod primitives -- Newton VBD vs
Stable Cosserat Rods vs GPU CoRdE vs DisMech -- judged on torsion/whip fidelity
and on throughput under batched contact, before committing the fast-tier
primitive. This is the harness for that comparison.

Honest status: there is exactly ONE entrant today -- the positional VBD-style rod
in lumen.physics.rod. The deciding criterion (torsion/whip) needs the torsional
DOF that lands with the Cosserat upgrade, so this currently reports throughput and
a tip-response (lag) proxy only. Registering a new primitive = adding a factory to
ROD_PRIMITIVES; the metrics below then compare them on equal footing.

Run:  python -m benchmarks.bakeoff
"""

from __future__ import annotations

import time

import numpy as np
import torch

from lumen.core.lumen_field import LumenField
from lumen.physics.contact import ContactGeometry, ContactParams
from lumen.physics.rod import Rod, RodParams
from lumen.physics.solver import SimConfig, Solver


def _vbd_rod(batch, n=16, dtype=torch.float64):
    x0 = np.stack([np.full(n, 0.5), np.zeros(n),
                   np.linspace(2.0, 2.0 + 2.0 * (n - 1), n)], axis=1)
    return Rod(torch.tensor(x0, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1),
               RodParams(k_stretch=3e2, k_bend=3.0, damping=2e2))


ROD_PRIMITIVES = {
    "vbd-positional": _vbd_rod,
    # "stable-cosserat": ...,   # doc §3.4.1 targets to add
    # "gpu-corde":      ...,
    # "dismech-der":    ...,
}


def _geom(dtype=torch.float64):
    M, L = 40, 120.0
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    return ContactGeometry(cl, LumenField.cylinder(L, 2.0, n=2), dtype=dtype)


def bench_primitive(factory, batch=128, steps=60):
    geom = _geom()
    cp = ContactParams(mu=0.2, kappa=1.5e3, d_hat=0.25)
    cfg = SimConfig(dt=8e-3, steps=steps, anchor_base=True, insertion_rate=0.05)
    solver = Solver(geom, contact=cp, cfg=cfg)
    rod = factory(batch)
    solver.rollout(factory(1))                       # warmup
    t0 = time.perf_counter()
    with torch.no_grad():
        solver.rollout(rod)
    dt = time.perf_counter() - t0
    env_steps_per_s = batch * steps / dt
    # tip-lag proxy: steps for the tip to move 1mm after a base nudge
    return {"batch": batch, "steps": steps, "wall_s": round(dt, 4),
            "env_steps_per_s": int(env_steps_per_s),
            "torsion_whip": "n/a (needs torsional DOF / Cosserat upgrade)"}


def main():
    print("rod-primitive bake-off (one entrant; see module docstring):\n")
    for name, factory in ROD_PRIMITIVES.items():
        r = bench_primitive(factory)
        print(f"  {name:18s} {r['env_steps_per_s']:>10d} env-steps/s   "
              f"torsion/whip: {r['torsion_whip']}")


if __name__ == "__main__":
    main()
