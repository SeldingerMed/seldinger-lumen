"""L0d.2d — two-way coaxial coupling: the catheter responds to the guidewire.

One-way leaves the catheter rigid (only the gw moves); two-way deposits the
equal-opposite reaction onto the catheter, so an offset guidewire pulls the catheter
toward it (a responsive catheter, the Newton-third-law fix to the one-way model)."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.newton.sim import NewtonGuidewireSim


def _wide_vessel(M=40, L=80.0):
    return np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)


def _line(n, x, z0, sp=2.0):
    return np.stack([np.full(n, x), np.zeros(n), z0 + np.arange(n) * sp], axis=1)


def _cath_lateral(two_way):
    # guidewire offset (+x 0.5) from the catheter axis (x=0), nested, wide vessel so only
    # the coupling acts. Measure how far the catheter is pulled toward the guidewire.
    sim = NewtonGuidewireSim(_wide_vessel(), 5.0, _line(11, 0.5, 2.0), radius=0.2,
                             catheter_points=_line(13, 0.0, 0.0), catheter_radius=0.4,
                             catheter_inner_radius=0.3, couple_coaxial=True,
                             coax_two_way=two_way, coax_kappa=5e3, vbd_iterations=12,
                             device="cpu")
    for _ in range(80):
        sim.step(dt=2.5e-2, substeps=5)
    cath = sim.catheter_positions()
    return float(np.abs(cath[:, 0]).max())          # catheter lateral excursion from x=0


def test_two_way_makes_the_catheter_respond_to_the_guidewire():
    two = _cath_lateral(two_way=True)
    one = _cath_lateral(two_way=False)
    assert one < 0.05                               # one-way: catheter stays on its axis
    assert two > 0.1                                # two-way: gw pulls the catheter toward it
    assert two > 3.0 * one


def test_two_way_assembly_is_stable_and_default_on():
    sim = NewtonGuidewireSim(_wide_vessel(), 5.0, _line(11, 0.4, 2.0),
                             catheter_points=_line(11, 0.0, 0.0), device="cpu")
    assert sim.solver._coax_two_way == 1            # two-way is the default
    for _ in range(40):
        sim.step(dt=2.5e-2, substeps=5, insertion=1.0)
    assert np.isfinite(sim.body_positions()).all()
    assert np.isfinite(sim.catheter_positions()).all()
