"""L0d.2a — coaxial assemblies: a microcatheter rod alongside the guidewire.

Two rods share the lumen with INDEPENDENT proximal actuation; here they interact
only through the shared wall contact (the sliding gw-in-catheter coupling is L0d.2b)."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.newton.sim import NewtonGuidewireSim


def _vessel(M=40, L=80.0):
    return np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)


def _rod(n, z0, x=0.3, sp=2.0):
    return np.stack([np.full(n, x), np.zeros(n), z0 + np.arange(n) * sp], axis=1)


def _coaxial(**kw):
    # catheter proximal (z 0..16), guidewire distal (z 18..36) — telescoping tandem
    return NewtonGuidewireSim(_vessel(), 2.0, _rod(10, 18.0), radius=0.2,
                              catheter_points=_rod(9, 0.0), catheter_radius=0.65,
                              vbd_iterations=10, device="cpu", **kw)


def test_coaxial_builds_with_two_rods():
    sim = _coaxial()
    assert sim.coaxial
    assert len(sim.bodies) == 9 and len(sim.cath_bodies) == 8     # 10/9 points -> 9/8 bodies
    assert sim.body_positions().shape == (9, 3)
    assert sim.catheter_positions().shape == (8, 3)
    sim.step(dt=1.5e-2, substeps=3)
    assert np.isfinite(sim.body_positions()).all()
    assert np.isfinite(sim.catheter_positions()).all()


def test_independent_actuation_of_each_device():
    sim = _coaxial()
    gw0 = sim.body_positions()[-1, 2]      # guidewire tip z
    ct0 = sim.catheter_positions()[-1, 2]  # catheter tip z
    for _ in range(10):                    # advance the CATHETER only
        sim.step(dt=2.5e-2, substeps=5, insertion_cath=2.0)
    assert sim.catheter_positions()[-1, 2] > ct0 + 3.0           # catheter advanced
    assert abs(sim.body_positions()[-1, 2] - gw0) < 2.0          # guidewire ~unchanged

    sim2 = _coaxial()
    gw0 = sim2.body_positions()[-1, 2]
    ct0 = sim2.catheter_positions()[-1, 2]
    for _ in range(10):                    # advance the GUIDEWIRE only
        sim2.step(dt=2.5e-2, substeps=5, insertion=2.0)
    assert sim2.body_positions()[-1, 2] > gw0 + 3.0             # guidewire advanced
    assert abs(sim2.catheter_positions()[-1, 2] - ct0) < 2.0    # catheter ~unchanged


def test_both_rods_held_in_lumen():
    sim = _coaxial()
    for _ in range(80):                    # press both against the wall
        sim.step(dt=2.5e-2, substeps=5, preload=(100.0, 0.0, 0.0))
    R = 2.0
    assert sim.node_radii().max() <= R + 0.3 + 0.1             # guidewire held
    assert sim.catheter_node_radii().max() <= R + 0.3 + 0.1    # catheter held


def test_coaxial_rejects_batched():
    with pytest.raises(NotImplementedError, match="single-env"):
        NewtonGuidewireSim(_vessel(), 2.0, _rod(10, 18.0), catheter_points=_rod(9, 0.0),
                           n_envs=2, device="cpu")


def test_coaxial_wires_thrombectomy_flow_clot_and_stentriever():
    from lumen.newton.devices import Stentriever
    from lumen.newton.flow import FlowField, FlowFieldParams
    sim = NewtonGuidewireSim(_vessel(M=60, L=120.0), 2.0, _rod(11, 40.0), radius=0.2,
                             catheter_points=_rod(13, 34.0), catheter_radius=0.65,
                             catheter_inner_radius=0.5,
                             flow=FlowField(FlowFieldParams(P_pulse=0.0)),
                             clot_segment=(55.0, 70.0), clot_height=1.2,
                             stentriever=Stentriever(deployed_center=62.0),
                             device="cpu")
    sim.step(dt=2.5e-2, substeps=2, aspiration=0.4)
    assert np.isfinite(sim.body_positions()).all()
    assert np.isfinite(sim.catheter_positions()).all()
    assert sim.flow.pressure_field() is not None
    assert sim.clot.o.max() > 0.0


def test_no_catheter_is_backward_compatible():
    sim = NewtonGuidewireSim(_vessel(), 2.0, _rod(11, 4.0), device="cpu")
    assert not sim.coaxial and sim.cath_bodies == []
    assert sim.catheter_positions().shape == (0, 3)
    sim.step(dt=1.5e-2, substeps=3)
    assert np.isfinite(sim.body_positions()).all()


def test_coaxial_with_deformable_wall():
    # GLM L1 / CodeRabbit #21: coaxial + deformable_wall is allowed (both rods press the
    # same single-env vessel wall) — verify it deflects and stays finite, not rejected.
    from lumen.newton.hgo_wall import HGOParams
    sim = NewtonGuidewireSim(_vessel(), 2.0, _rod(10, 18.0), radius=0.2,
                             catheter_points=_rod(9, 0.0), catheter_radius=0.65,
                             deformable_wall=True,
                             hgo_params=HGOParams(C10=3e3, k1=1.5e3, k2=1.0, thickness=0.3),
                             device="cpu")
    for _ in range(60):
        sim.step(dt=2.5e-2, substeps=5, preload=(120.0, 0.0, 0.0))
    assert np.isfinite(sim.body_positions()).all() and np.isfinite(sim.catheter_positions()).all()
    assert sim.wall_max_deflection() > 1e-4        # both rods deform the shared wall


def test_degenerate_catheter_rejected():
    # CodeRabbit #22: a < 2-node catheter centerline can't define a rod -> fail fast up front
    with pytest.raises(ValueError, match=">= 2 nodes"):
        NewtonGuidewireSim(_vessel(), 2.0, _rod(10, 18.0), catheter_points=_rod(1, 0.0), device="cpu")


def test_guidewire_too_thick_for_catheter_rejected():
    # CodeRabbit #6: a guidewire that can't fit inside the catheter inner lumen is a hard
    # error, not a near-singular clamp (gw radius >= inner radius).
    with pytest.raises(ValueError, match="must be <"):
        NewtonGuidewireSim(_vessel(), 2.0, _rod(10, 18.0), radius=0.5,
                           catheter_points=_rod(9, 0.0), catheter_inner_radius=0.5, device="cpu")
