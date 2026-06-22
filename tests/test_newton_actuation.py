"""On-device proximal actuation (actuate_bases kernel): insertion translates the
kinematic base along its current axis; twist spins it about that axis."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")


def _sim():
    from lumen.newton.sim import NewtonGuidewireSim
    M, L, R, n = 30, 60.0, 2.0, 9
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.zeros(n), np.zeros(n), np.linspace(4, 4 + 2 * (n - 1), n)], axis=1)
    return NewtonGuidewireSim(vessel, R, dev, radius=0.2, device="cpu")


def test_insertion_advances_base_along_axis():
    sim = _sim()
    base0 = sim.body_positions()[0].copy()           # kinematic base, +z axis
    sim.step(dt=2.5e-2, substeps=4, insertion=2.0)    # total +2 along z
    base1 = sim.body_positions()[0]
    dz = base1[2] - base0[2]
    assert abs(dz - 2.0) < 1e-4                        # moved by the commanded insertion
    assert abs(base1[0] - base0[0]) < 1e-4 and abs(base1[1] - base0[1]) < 1e-4


def test_zero_action_is_a_noop_for_the_base():
    sim = _sim()
    base0 = sim.body_positions()[0].copy()
    sim.step(dt=2.5e-2, substeps=4, insertion=0.0, twist=0.0)
    assert np.allclose(sim.body_positions()[0], base0, atol=1e-6)
