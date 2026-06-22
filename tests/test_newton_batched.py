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


def test_batched_deformable_wall_is_per_env():
    # Each env has its OWN HGO wall block; loading one env's block must not deflect
    # another's (no cross-env bleed in the per-cell solve or the shell smoothing).
    from lumen.newton.hgo_wall import WallField, HGOParams
    E, n_s, n_th = 3, 20, 8
    ncell = n_s * n_th
    w = WallField(R0=2.0, s_max=80.0, n_s=n_s, n_th=n_th, n_envs=E, device="cpu",
                  params=HGOParams(C10=2e3, k1=1e3, k2=1.0, thickness=0.3))
    load = np.zeros(E * ncell, np.float32)
    mid = (n_s // 2) * n_th
    load[1 * ncell + mid:1 * ncell + mid + n_th] = 50.0          # load only env 1
    w.wall_load.assign(load)
    w.update_from_load()
    wf = w.w_field.numpy().reshape(E, ncell)
    assert wf[1].max() > 1e-3                                     # the loaded env deflects
    assert wf[0].max() < 1e-9 and wf[2].max() < 1e-9             # the others do not


def test_batched_deformable_sim_runs_and_stays_finite():
    vessel, dev = _vessel_and_device(n=11)
    dev = dev.copy(); dev[:, 0] = 1.85                           # start the wire in contact
    E = 3
    sim = NewtonGuidewireSim(vessel, 2.0, dev, radius=0.2, kappa=3e3, d_hat=0.3,
                             deformable_wall=True, n_envs=E, vbd_iterations=12,
                             device="cpu")
    for _ in range(20):
        sim.step(dt=2.5e-2, substeps=5, preload=(40.0, 0.0, 0.0))
    w = sim.solver._wall
    assert w.w_field.shape[0] == E * w.n_cells                   # one wall block per env
    assert np.isfinite(sim.env_positions()).all()
    assert w.w_field.numpy().max() > 1e-3                        # the deformable walls deflected


def test_batched_clot_update_is_per_env():
    # The on-device clot kernel evolves each env's occlusion independently: loading
    # only one env's clot block compresses/damages just that env.
    import warp as wp
    from lumen.newton.clot import ClotField, ClotParams
    E, n_s, n_th = 3, 40, 8
    ncell = n_s * n_th
    c = ClotField(80.0, n_s, n_th, 2.0, 35, 45, 1.6, ClotParams(), n_envs=E, device="cpu")
    load = np.zeros(E * ncell, np.float32)
    s = np.linspace(0, 80, n_s); mask = (s >= 35) & (s <= 45)
    for i in np.where(mask)[0]:
        load[1 * ncell + i * n_th:1 * ncell + (i + 1) * n_th] = 0.05   # only env 1
    wl = wp.array(load, dtype=wp.float32, device="cpu")
    for _ in range(20):
        c.update_device(wl, dt=1e-2)
    o = c.o_d.numpy().reshape(E, n_s)
    assert o[0].max() > 1.5 and o[2].max() > 1.5        # untouched envs keep their clot
    assert o[1].max() < 0.5                              # the loaded env's clot collapsed


def test_batched_clot_and_flow_run_on_device():
    # Batched clot + 1-D FlowField run through the on-device coupling path (no host
    # round-trip per substep) and stay finite, with per-env wall/flow blocks.
    from lumen.newton.flow import FlowField
    vessel, dev = _vessel_and_device(M=60, L=120.0, n=11)
    E = 3
    flow = FlowField()
    sim = NewtonGuidewireSim(vessel, 2.0, dev, radius=0.2, kappa=3e3, d_hat=0.3,
                             flow=flow, clot_segment=(55, 70), clot_height=1.6,
                             n_envs=E, device="cpu")
    assert sim._use_device_coupling and flow.n_envs == E
    for _ in range(8):
        sim.step(dt=2.5e-2, substeps=2, insertion=np.array([0.5, 1.0, 1.5]))
    w = sim.solver._wall
    assert flow._v_d.shape[0] == E * w.n_s              # a velocity field per env
    assert sim.clot.o_d.shape[0] == E * w.n_s           # a clot field per env
    assert np.isfinite(sim.env_positions()).all()


def test_flowfield_not_shared_across_sims():
    # a FlowField sizes its device arrays to n_envs; binding a conflicting or
    # already-used one to a sim must be refused (defensive against silent shape bugs)
    from lumen.newton.flow import FlowField
    vessel, dev = _vessel_and_device()
    f = FlowField(); f.n_envs = 2
    with pytest.raises(ValueError):                  # n_envs conflict
        NewtonGuidewireSim(vessel, 2.0, dev, flow=f, n_envs=3, device="cpu")
    f2 = FlowField()
    s1 = NewtonGuidewireSim(vessel, 2.0, dev, flow=f2, clot_segment=(30, 40),
                            n_envs=2, device="cpu")
    s1.step(dt=2.5e-2, substeps=1, insertion=np.array([0.3, 0.3]))   # allocates f2 device arrays
    with pytest.raises(ValueError):                  # already bound/used
        NewtonGuidewireSim(vessel, 2.0, dev, flow=f2, n_envs=2, device="cpu")


def test_batched_rejects_lumped_flow_and_stentriever():
    vessel, dev = _vessel_and_device()
    from lumen.newton.flow import NewtonFlow
    from lumen.newton.devices import Stentriever
    with pytest.raises(NotImplementedError):            # lumped flow is single-env only
        NewtonGuidewireSim(vessel, 2.0, dev, n_envs=2, flow=NewtonFlow(), device="cpu")
    with pytest.raises(NotImplementedError):            # batched retrieval not ported
        NewtonGuidewireSim(vessel, 2.0, dev, n_envs=2,
                           stentriever=Stentriever(deployed_center=40.0, span=10.0),
                           device="cpu")
