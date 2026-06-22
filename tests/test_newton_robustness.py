"""Contact/actuation robustness: fast proximal insertion into a CURVED vessel must
keep the wire inside the lumen (the kinematic base follows the centerline arc-length,
not a straight line that drifts out the outer wall of a bend)."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.newton.sim import NewtonGuidewireSim


def _curved_vessel(R=2.0, n=11):
    a = np.linspace(0, np.pi / 2, 80)
    cl = np.stack([60 * np.sin(a), np.zeros_like(a), 60 * (1 - np.cos(a))], axis=1)  # 90° bend
    dev = cl[:n].copy()                              # wire seeded conforming, in the lumen
    return cl, R, dev


def test_fast_insertion_through_curve_stays_in_lumen():
    cl, R, dev = _curved_vessel()
    # fast insertion (0.5/step) that previously drove the kinematic base straight
    # through the outer wall (penetration ~7); centerline-following keeps it bounded.
    sim = NewtonGuidewireSim(cl, R, dev, radius=0.2, kappa=3e3, d_hat=0.3,
                             bend_stiffness=2e2, mu_along=0.1, mu_across=0.4,
                             gamma_fric_deg=30, vbd_iterations=14, device="cpu")
    peak_pen = 0.0
    for _ in range(70):
        sim.step(dt=2.5e-2, substeps=5, insertion=0.5)
        peak_pen = max(peak_pen, float((sim.node_radii() - R).max()))
    assert np.isfinite(sim.body_positions()).all()
    assert peak_pen < 0.3                             # within the d_hat contact band, not r~7
    # the wire actually advanced along the curve (insertion did something)
    tip_s = sim.contact_frame.project(sim.body_positions()[-1]).s
    assert tip_s > 30.0


def test_insertion_advances_along_curved_arclength():
    # the base rides the centerline: its arc-length grows ~linearly with insertion,
    # and it stays on the centerline (near-zero radius) through the bend
    cl, R, dev = _curved_vessel()
    sim = NewtonGuidewireSim(cl, R, dev, radius=0.2, kappa=3e3, d_hat=0.3, device="cpu")
    s_prev = sim.contact_frame.project(sim.body_positions()[0]).s
    for _ in range(30):
        sim.step(dt=2.5e-2, substeps=4, insertion=0.4)
    s_now = sim.contact_frame.project(sim.body_positions()[0]).s
    assert s_now - s_prev > 3.0                       # base advanced along the arc
    assert sim.contact_frame.project(sim.body_positions()[0]).r < 0.5   # stayed on centerline
