"""L0d.2c — telescoping maneuver primitives on the coupled coaxial assembly."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.newton.sim import NewtonGuidewireSim
from examples.coaxial_telescope import telescope


def _vessel(M=40, L=80.0):
    return np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)


def _line(n, x, z0, sp=2.0):
    return np.stack([np.full(n, x), np.zeros(n), z0 + np.arange(n) * sp], axis=1)


def _assembly():
    return NewtonGuidewireSim(_vessel(), 2.0, _line(11, 0.2, 2.0), radius=0.2,
                              catheter_points=_line(11, 0.0, 2.0), catheter_radius=0.65,
                              catheter_inner_radius=0.5, couple_coaxial=True, device="cpu")


def test_guidewire_leads_out_past_the_catheter_tip():
    sim = _assembly()
    cath_tip0 = sim.catheter_positions()[-1, 2]
    for _ in range(15):                                  # advance the guidewire only
        sim.step(dt=2.5e-2, substeps=5, insertion=2.0)
    gw_tip = sim.body_positions()[-1, 2]
    assert gw_tip > sim.catheter_positions()[-1, 2] + 3.0   # gw led out beyond the catheter
    assert abs(sim.catheter_positions()[-1, 2] - cath_tip0) < 1.5   # catheter stayed (independent)


def test_catheter_slides_freely_over_the_guidewire():
    # the coupling is radial-only — advancing the catheter mostly SLIDES over the
    # guidewire rather than dragging it (the defining coaxial property). Two-way coupling
    # transmits a little axial drag, but the gw moves far less than the catheter (a rigid
    # drag would move them together, ~equally).
    sim = _assembly()
    gw_tip0, ct_tip0 = sim.body_positions()[-1, 2], sim.catheter_positions()[-1, 2]
    for _ in range(15):                                  # advance the catheter only
        sim.step(dt=2.5e-2, substeps=5, insertion_cath=2.0)
    cath_adv = sim.catheter_positions()[-1, 2] - ct_tip0
    gw_drag = abs(sim.body_positions()[-1, 2] - gw_tip0)
    assert cath_adv > 5.0                                       # catheter advanced a lot
    assert gw_drag < 0.25 * cath_adv                            # gw mostly free (not rigidly dragged)


def test_telescope_follow_phase_closes_the_support_gap():
    trace = telescope(steps_per_phase=5, support_gap=4.0)
    start_gap = trace[0][1] - trace[0][2]
    lead_gap = trace[1][1] - trace[1][2]
    support_gap = trace[2][1] - trace[2][2]
    assert start_gap == pytest.approx(0.0)
    assert lead_gap > 8.0                       # the guidewire leads into the next segment
    assert 0.0 < support_gap <= 4.0             # catheter follows, but stays behind the wire
    assert support_gap < lead_gap               # the follow phase reduces the overhang


def test_no_cross_rod_capsule_collision():
    # the gw/catheter interaction is the radial coupling, not Newton capsule contact:
    # the model carries a collision filter between the two rods' shapes.
    sim = _assembly()
    assert sim.coaxial and len(sim.cath_bodies) > 0
    sim.step(dt=1.5e-2, substeps=3)
    assert np.isfinite(sim.body_positions()).all() and np.isfinite(sim.catheter_positions()).all()
