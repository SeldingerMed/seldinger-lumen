"""Batched multi-env execution: E independent guidewires in one Newton model/solver
(shared vessel; contact is wire-vs-wall, never wire-vs-wire), driven by per-env
actions. This is the GPU-throughput mechanism for RL."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.newton.sim import NewtonGuidewireSim


def _vessel_and_device(M=30, L=60.0, n=9):
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.zeros(n), np.zeros(n), np.linspace(4, 4 + 2 * (n - 1), n)], axis=1)
    return vessel, dev


def test_batched_envs_are_independent_under_per_env_actions():
    vessel, dev = _vessel_and_device()
    E = 4
    sim = NewtonGuidewireSim(vessel, 2.0, dev, radius=0.2, n_envs=E, device="cpu")
    assert len(sim.bodies) == E * sim.n_per_env
    assert sim.bases == [e * sim.n_per_env for e in range(E)]

    ins = np.array([0.0, 1.0, 2.0, 3.0])           # a different insertion per env
    for _ in range(5):
        sim.step(dt=2.5e-2, substeps=3, insertion=ins)

    ep = sim.env_positions()
    assert ep.shape == (E, sim.n_per_env, 3)
    assert np.isfinite(ep).all()
    tip_z = ep[:, -1, 2]
    # more insertion -> deeper tip, strictly ordered: the envs really are independent
    assert np.all(np.diff(tip_z) > 0.5)


def test_scalar_action_broadcasts_to_all_envs():
    vessel, dev = _vessel_and_device()
    sim = NewtonGuidewireSim(vessel, 2.0, dev, radius=0.2, n_envs=3, device="cpu")
    for _ in range(4):
        sim.step(dt=2.5e-2, substeps=3, insertion=1.5)   # same action for all envs
    tip_z = sim.env_positions()[:, -1, 2]
    assert np.allclose(tip_z, tip_z[0], atol=1e-4)        # identical -> all envs match


def test_batched_rejects_unported_per_env_features():
    vessel, dev = _vessel_and_device()
    with pytest.raises(NotImplementedError):
        NewtonGuidewireSim(vessel, 2.0, dev, n_envs=2, deformable_wall=True, device="cpu")
