"""L0d.2b — sliding coaxial coupling: the guidewire rides inside the catheter lumen.

Seeds the guidewire OFFSET from the catheter axis (outside the inner lumen) in a wide
vessel (so wall contact doesn't interfere) and checks the coupling pulls the free gw
into the catheter, while the same scene WITHOUT coupling leaves it offset."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.newton.sim import NewtonGuidewireSim


def _wide_vessel(M=40, L=80.0):
    return np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)


def _line(n, x, z0, sp=2.0):
    return np.stack([np.full(n, x), np.zeros(n), z0 + np.arange(n) * sp], axis=1)


def _point_to_polyline(p, poly):
    """Min distance from point p to the polyline `poly` (segment-wise)."""
    best = np.inf
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        u = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
        best = min(best, np.linalg.norm(p - (a + u * ab)))
    return best


def _run(couple):
    # catheter on the axis (x=0); guidewire offset by 0.5 (> r_inner=0.3) in a WIDE
    # vessel (R=5) so the only thing acting on the gw radially is the coupling.
    # catheter spans z 0..24 (longer) so the guidewire (z 2..22) is FULLY nested — no
    # tip-overhang; the only gw–catheter distance is the radial offset.
    sim = NewtonGuidewireSim(_wide_vessel(), 5.0, _line(11, 0.5, 2.0), radius=0.2,
                             catheter_points=_line(13, 0.0, 0.0), catheter_radius=0.4,
                             catheter_inner_radius=0.3, couple_coaxial=couple,
                             coax_kappa=5e3, vbd_iterations=12, device="cpu")
    for _ in range(80):
        sim.step(dt=2.5e-2, substeps=5)
    cath = sim.catheter_positions()
    tip = sim.body_positions()[-1]                 # the free guidewire tip (nested in the catheter)
    return _point_to_polyline(tip, cath)


def test_coupling_pulls_guidewire_into_the_catheter():
    coupled = _run(couple=True)
    free = _run(couple=False)
    assert free > 0.4                              # without coupling the gw tip stays offset
    assert coupled < 0.3 + 0.12                    # coupling pulls it inside the inner lumen
    assert coupled < free - 0.1                    # decisive difference


def test_coupling_can_be_disabled():
    sim = NewtonGuidewireSim(_wide_vessel(), 5.0, _line(11, 0.5, 2.0),
                             catheter_points=_line(11, 0.0, 0.0), couple_coaxial=False,
                             device="cpu")
    assert not getattr(sim.solver, "_coax_enabled", False)
    sim.step(dt=1.5e-2, substeps=3)
    assert np.isfinite(sim.body_positions()).all()


def test_coupled_assembly_is_stable():
    sim = NewtonGuidewireSim(_wide_vessel(), 5.0, _line(11, 0.4, 2.0),
                             catheter_points=_line(11, 0.0, 0.0), couple_coaxial=True,
                             device="cpu")
    for _ in range(40):
        sim.step(dt=2.5e-2, substeps=5, insertion=1.0)   # advance the coupled assembly
    assert np.isfinite(sim.body_positions()).all()
    assert np.isfinite(sim.catheter_positions()).all()
