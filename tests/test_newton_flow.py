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
    with pytest.raises(ValueError, match=r"tree tip edge_index values must be in \[0, 3\)"):
        f.set_tree_tips([0, 3], [1.0, 2.0])


def test_tree_flow_rejects_unsolved_drag_query():
    f = FlowField()
    f.set_tree_lumen(np.ones((1, 1, 3)), np.ones(1))
    with pytest.raises(RuntimeError, match=r"solve_tree\(\) must be called before drag_at_tree\(\)"):
        f.drag_at_tree([0], [0], [0.5])


def test_tree_flow_invalidates_solved_fields_when_inputs_change():
    f = FlowField()
    f.set_tree_lumen(np.ones((1, 1, 3)), np.ones(1))
    f.set_tree_tips([0], [0.5])
    f.solve_tree()
    assert f.tree_velocity_fields() is not None

    f.set_tree_lumen(np.full((1, 1, 3), 2.0), np.ones(1))
    assert f.tree_velocity_fields() is None
    assert f._tree_tip_edge is None
    assert f._tree_tip_s is None
    with pytest.raises(RuntimeError, match=r"solve_tree\(\) must be called before drag_at_tree\(\)"):
        f.drag_at_tree([0], [0], [0.5])

    f.set_tree_tips([0], [0.25])
    f.solve_tree()
    f.set_tree_tips([0], [0.75])
    assert f.tree_velocity_fields() is None


def test_tree_flow_lumen_shape_change_drops_stale_tips():
    f = FlowField()
    f.set_tree_lumen(np.ones((1, 1, 3)), np.ones(1))
    f.set_tree_tips([0], [0.5])
    f.set_tree_lumen(np.ones((2, 1, 3)), np.ones(1))
    f.solve_tree()
    qdown = f.tree_downstream_Q()
    assert qdown is not None
    assert qdown.shape == (2, 1)


def test_tree_drag_matches_linear_edge_interpolation():
    f = FlowField()
    f.set_tree_lumen(np.ones((2, 2, 3)), np.array([2.0, 4.0]))
    f._tree_v = np.array([
        [[0.0, 2.0, 4.0], [10.0, 14.0, 18.0]],
        [[1.0, 3.0, 5.0], [20.0, 28.0, 36.0]],
    ])
    drag = f.drag_at_tree(
        np.array([[0, 1], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1.0, 1.0], [2.0, 2.0]]),
    )
    expected_v = np.array([[2.0, 3.0], [14.0, 28.0]])
    assert np.allclose(drag, f.p.drag_coeff * expected_v)


def test_tree_drag_shape_validation_is_explicit():
    f = FlowField()
    f.set_tree_lumen(np.ones((2, 3, 5)), np.ones(3))
    f.solve_tree()
    with pytest.raises(ValueError, match="tree drag inputs must have matching shapes"):
        f.drag_at_tree([0, 1], [0], [0.0, 1.0])


def test_tree_pressure_field_assembled_without_overlap_at_boundaries():
    """P(e, g) must be well-defined when the tip sits at the inlet (it==0) or
    distal end (it==S-1); the previous version wrote overlapping slices and
    relied on write order to land the right value, which is brittle when the
    assembly is refactored. Pin the assembled field at both edges."""
    f = FlowField()
    f.set_tree_lumen(np.ones((1, 1, 4)), np.array([3.0]))
    P_grid = f.tree_pressure_fields()        # not solved yet
    assert P_grid is None
    f.set_tree_tips([0], [0.0])              # tip at the inlet: it = 0
    f.solve_tree()
    P_inlet = f.tree_pressure_fields()
    assert P_inlet.shape == (1, 1, 4)
    assert np.all(np.isfinite(P_inlet))
    assert np.allclose(P_inlet[0, 0], P_inlet[0, 0, 0])   # constant when R_up == 0

    f.set_tree_tips([0], [3.0])              # tip at the distal end: it = S - 1
    f.solve_tree()
    P_distal = f.tree_pressure_fields()
    assert P_distal.shape == (1, 1, 4)
    assert np.all(np.isfinite(P_distal))
    # Downstream pressure at the tip index must be a well-defined finite scalar
    # and the field must be monotonically non-increasing on a uniform tube
    # (pressure drops as the wall is crossed tip-to-inlet).
    assert np.isfinite(float(P_distal[0, 0, -1]))
    assert np.all(np.diff(P_distal[0, 0]) <= 0)             # monotonic drop on a uniform tube


def test_tree_solve_handles_single_sample_edges():
    """Degenerate edge (S==1) takes the explicit degenerate branch and still
    produces a finite pressure, velocity, and downstream Q without dividing by
    the empty r_mid array."""
    f = FlowField()
    f.set_tree_lumen(np.ones((1, 1, 1)), np.array([2.0]))
    f.set_tree_tips([0], [0.0])
    f.solve_tree()
    P = f.tree_pressure_fields()
    v = f.tree_velocity_fields()
    qdown = f.tree_downstream_Q()
    assert P.shape == (1, 1, 1)
    assert v.shape == (1, 1, 1)
    assert qdown.shape == (1, 1)
    assert np.all(np.isfinite(P))
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(qdown))


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
