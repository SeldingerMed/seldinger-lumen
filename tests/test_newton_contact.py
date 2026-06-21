"""Task #1: guidewire on Newton VBD with tube-intrinsic contact in the AVBD solve.

Skipped if `newton` is not installed (it is an optional, heavyweight backend).
Runs on the Warp CPU device locally and on CUDA where present.
"""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.newton.sim import NewtonGuidewireSim


def _vessel(M=40, L=80.0):
    return np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)


def _device_points(n=11, x=1.0):
    return np.stack([np.full(n, x), np.zeros(n), np.linspace(4, 4 + 2.0 * (n - 1), n)],
                    axis=1)


def _run(enable_contact, preload=120.0, nsteps=200):
    R = 2.0
    sim = NewtonGuidewireSim(_vessel(), R, _device_points(), radius=0.2,
                             stretch_stiffness=1e4, bend_stiffness=4e1,
                             kappa=2e3, d_hat=0.3, vbd_iterations=10, device="cpu")
    if not enable_contact:
        sim.solver._tube_enabled = False        # disable the injected barrier
    for _ in range(nsteps):
        sim.step(dt=5e-3, substeps=5, preload=(preload, 0.0, 0.0))
    return sim.node_radii(), R


def test_guidewire_builds_and_steps_on_newton():
    sim = NewtonGuidewireSim(_vessel(), 2.0, _device_points())
    sim.step(dt=5e-3, substeps=3)
    assert np.isfinite(sim.body_positions()).all()
    assert len(sim.bodies) == 10        # 11 centerline points -> 10 capsule bodies


def test_tube_contact_holds_device_in_lumen_vs_escapes_without():
    r_off, R = _run(enable_contact=False)
    r_on, _ = _run(enable_contact=True)
    # without contact a strong preload flings the wire far outside the lumen...
    assert r_off.max() > 2.0 * R
    # ...with the implicit barrier it is held within the barrier band (R + d_hat)
    assert r_on.max() <= R + 0.3 + 0.05
    assert np.isfinite(r_on).all()
    # and the barrier makes a large, decisive difference
    assert r_on.max() < 0.3 * r_off.max()
