"""Task: flow on Newton — Windkessel, pulsatility, drag, two-way occlusion/aspiration."""

import math

import numpy as np
import pytest

from lumen.newton.flow import NewtonFlow, FlowParams, FlowField


def test_windkessel_decay_and_pulse():
    f = NewtonFlow(FlowParams(R_periph=1.0, C=1.5, Q_mean=4.0, Q_pulse=2.0, heart_rate=1.0))
    assert abs(f.pressure_decay(100.0, 0.3) - 100 * math.exp(-0.3 / 1.5)) < 1e-9
    peak = max(f.Q(t * 0.01) for t in range(100))
    assert 4.0 < peak <= 6.0 + 1e-9                       # pulsatile, peaks above mean


def test_downstream_Q_two_way():
    f = NewtonFlow(FlowParams(Q_mean=4.0, Q_pulse=0.0))
    assert abs(f.downstream_Q() - 4.0) < 1e-9            # no occlusion -> full flow
    f.occlusion = 0.9
    assert f.downstream_Q() < 0.5                         # clot occludes downstream flow
    f.aspiration = 0.5
    assert 0.5 < f.downstream_Q() < 4.0                   # aspiration recovers part
    f.occlusion = 5.0; f.aspiration = 5.0                 # out-of-range
    assert 0.0 <= f.downstream_Q() <= 4.0 + 1e-9         # bounded


def test_drag_scales_with_flow():
    lo = NewtonFlow(FlowParams(Q_mean=2.0, Q_pulse=0.0, drag_coeff=30.0))
    hi = NewtonFlow(FlowParams(Q_mean=6.0, Q_pulse=0.0, drag_coeff=30.0))
    assert hi.drag_per_unit_tangent() > lo.drag_per_unit_tangent() > 0


def test_tree_flow_tip_shape_validation_is_explicit():
    f = FlowField()
    f.set_tree_lumen(np.ones((2, 3, 5)), np.ones(3))
    with pytest.raises(ValueError, match=r"tree tip edge_index must broadcast to \(n_envs,\)"):
        f.set_tree_tips([0, 1, 2], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"tree tip s_tip must broadcast to \(n_envs,\)"):
        f.set_tree_tips([0, 1], [1.0, 2.0, 3.0])


def test_tree_drag_shape_validation_is_explicit():
    f = FlowField()
    f.set_tree_lumen(np.ones((2, 3, 5)), np.ones(3))
    f.solve_tree()
    with pytest.raises(ValueError, match="tree drag inputs must have matching shapes"):
        f.drag_at_tree([0, 1], [0], [0.0, 1.0])


def test_pulsatility_modulates_lumen_in_sim():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.newton.sim import NewtonGuidewireSim
    M, L, R, n = 40, 80.0, 2.0, 11
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.zeros(n), np.zeros(n), np.linspace(4, 4 + 2 * (n - 1), n)], axis=1)
    sim = NewtonGuidewireSim(vessel, R, dev, radius=0.2, kappa=3e3, d_hat=0.3,
                             flow=NewtonFlow(FlowParams(pulse_amp=0.1, heart_rate=2.0)),
                             device="cpu")
    r0s = []
    for _ in range(20):
        sim.step(dt=2.5e-2, substeps=2)
        r0s.append(float(sim.solver._wall.r0_field.numpy().mean()))
    assert max(r0s) - min(r0s) > 0.05                     # lumen R(s,θ,t) breathes with the cycle
    assert np.isfinite(sim.body_positions()).all()        # drag-coupled sim stays stable
