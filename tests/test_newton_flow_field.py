"""1-D resistive-network flow field (doc §3.4.3): emergent occlusion, velocity
jet, pressure drop localized at the clot, and aspiration as a mobilizing sink."""

import numpy as np
import pytest

from lumen.newton.flow import FlowField, FlowFieldParams


def _vessel(Rprofile):
    f = FlowField(FlowFieldParams(P_pulse=0.0))      # steady, for deterministic checks
    s = np.linspace(0, 80, 41)
    f.set_lumen(Rprofile(s), 80.0)
    f.solve()
    return f, s


def test_occlusion_drops_flow_emergently():
    # flow drop is a CONSEQUENCE of raised resistance, not a fed-in occlusion scalar
    f_open, _ = _vessel(lambda s: np.full_like(s, 2.0))
    f_clot, _ = _vessel(lambda s: np.where((s >= 35) & (s <= 45), 0.4, 2.0))
    assert f_clot.Q() < 0.1 * f_open.Q()             # narrowing throttles through-flow


def test_velocity_jets_at_narrowing():
    f, s = _vessel(lambda s: np.where((s >= 35) & (s <= 45), 0.4, 2.0))
    v = f.velocity_field()
    in_clot = (s >= 35) & (s <= 45)
    assert v[in_clot].max() > 5.0 * v[~in_clot].max()   # continuity -> jet through the throat


def test_pressure_drop_localizes_at_clot():
    f, s = _vessel(lambda s: np.where((s >= 35) & (s <= 45), 0.4, 2.0))
    P = f.pressure_field()
    drop_clot = P[s < 35][-1] - P[s > 45][0]            # ΔP across the clot
    drop_open = P[0] - P[s < 35][-1]                    # ΔP across the open proximal run
    assert drop_clot > 10.0 * drop_open                 # nearly all the loss is at the clot


def test_aspiration_flips_mobilizing_force():
    f, s = _vessel(lambda s: np.where((s >= 35) & (s <= 45), 0.4, 2.0))
    f.set_tip(34.0)                                      # catheter tip just proximal of clot
    f.aspiration = 0.0; f.solve()
    resist = f.clot_mobilizing_force(35, 45)
    f.aspiration = 1.0; f.solve()
    assist = f.clot_mobilizing_force(35, 45)
    assert resist < 0 < assist                          # antegrade resists, suction assists


def test_flow_field_drives_sim_with_local_drag():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.newton.sim import NewtonGuidewireSim
    M, L, R, n = 50, 100.0, 2.0, 11
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.zeros(n), np.zeros(n), np.linspace(4, 4 + 2 * (n - 1), n)], axis=1)
    sim = NewtonGuidewireSim(vessel, R, dev, radius=0.2, kappa=3e3, d_hat=0.3,
                             flow=FlowField(FlowFieldParams()), device="cpu")
    for _ in range(20):
        sim.step(dt=2.5e-2, substeps=2)
    assert np.isfinite(sim.body_positions()).all()      # field-flow-coupled sim is stable
    assert sim.flow.pressure_field() is not None         # the 1-D field actually solved
