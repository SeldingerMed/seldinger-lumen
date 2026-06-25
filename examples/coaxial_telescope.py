"""L0d.2c — a telescoping coaxial maneuver: support with the catheter, lead with the wire.

    python examples/coaxial_telescope.py

The clinical primitive: a microcatheter advances for support, then the guidewire is
pushed BEYOND the catheter tip to lead into the next segment — the guidewire sliding
inside the coupled catheter. Prints the two tip positions through the maneuver. Needs
newton + warp.
"""

from __future__ import annotations

import numpy as np

from lumen.newton.sim import NewtonGuidewireSim


def _vessel(M=40, L=80.0):
    return np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)


def _line(n, x, z0, sp=2.0):
    return np.stack([np.full(n, x), np.zeros(n), z0 + np.arange(n) * sp], axis=1)


def telescope(steps_per_phase=15):
    """Phase 1: lead the guidewire out past the catheter tip. Phase 2: advance the
    catheter to follow for support. Returns [(label, gw_tip_z, cath_tip_z), ...]."""
    # start roughly tip-aligned (gw inside the catheter)
    sim = NewtonGuidewireSim(_vessel(), 2.0, _line(11, 0.2, 2.0), radius=0.2,
                             catheter_points=_line(11, 0.0, 2.0), catheter_radius=0.4,
                             catheter_inner_radius=0.3, couple_coaxial=True, device="cpu")

    def tips():
        return float(sim.body_positions()[-1, 2]), float(sim.catheter_positions()[-1, 2])

    trace = [("start", *tips())]
    for _ in range(steps_per_phase):                 # phase 1: guidewire leads out
        sim.step(dt=2.5e-2, substeps=5, insertion=2.0)
    trace.append(("guidewire led out", *tips()))
    for _ in range(steps_per_phase):                 # phase 2: catheter follows (support)
        sim.step(dt=2.5e-2, substeps=5, insertion_cath=2.0)
    trace.append(("catheter advanced", *tips()))
    return trace


def main():
    print(f"{'phase':20s} {'gw_tip_z':>10s} {'cath_tip_z':>12s} {'gw beyond cath':>16s}")
    for name, gw, ct in telescope():
        print(f"{name:20s} {gw:10.2f} {ct:12.2f} {gw - ct:16.2f}")


if __name__ == "__main__":
    main()
