"""Task #4 (torsion): proximal rotation transmits twist to the distal tip.

Torque transmission / whip is the basis of steerability and a doc gate
(§3.5.5, §3.11). Newton's cable carries a bend/twist DOF, so spinning the
kinematic base body propagates twist down the wire. Skipped without `newton`.
"""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

import warp as wp
import newton


def _roll_z(quat):
    x, y, z, w = quat
    return 2.0 * np.arctan2(z, w)


# module-level so Warp can introspect the kernel source on a cold cache
@wp.kernel
def _spin(bid: int, rate: float, dt: float,
          q0: wp.array(dtype=wp.transform), q1: wp.array(dtype=wp.transform)):
    t = q0[bid]
    pos = wp.transform_get_translation(t)
    rot = wp.transform_get_rotation(t)
    ax = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    dq = wp.quat_from_axis_angle(ax, rate * dt)
    T = wp.transform(pos, wp.mul(dq, rot))
    q0[bid] = T
    q1[bid] = T


def test_twist_propagates_to_distal_tip():
    wp.init()
    b = newton.ModelBuilder(gravity=0.0)
    b.default_shape_cfg.density = 1.0
    n = 16
    pts = [wp.vec3(0.0, 0.0, float(z)) for z in np.linspace(0, 32, n + 1)]
    quats = newton.utils.create_parallel_transport_cable_quaternions(pts)
    bodies, _ = b.add_rod(pts, quats, radius=0.15, stretch_stiffness=1e6,
                          bend_stiffness=5e3, bend_damping=50.0,
                          body_frame_origin="com")
    base = bodies[0]
    b.body_mass[base] = 0.0
    b.body_inv_mass[base] = 0.0
    b.body_inertia[base] = wp.mat33(0.0)
    b.body_inv_inertia[base] = wp.mat33(0.0)
    b.color()
    model = b.finalize(device="cpu")
    solver = newton.solvers.SolverVBD(model, iterations=8)
    s0, s1 = model.state(), model.state()
    c, ct = model.control(), model.contacts()

    spin = _spin
    tip0 = _roll_z(s0.body_q.numpy()[bodies[-1]][3:7])
    for _ in range(120):
        for _ in range(10):
            s0.clear_forces()
            wp.launch(spin, dim=1, inputs=[base, 3.0, 2e-3, s0.body_q, s1.body_q])
            solver.step(s0, s1, c, ct, 2e-3)
            s0, s1 = s1, s0
    q = s0.body_q.numpy()
    tip_roll = abs(_roll_z(q[bodies[-1]][3:7]) - tip0)
    assert np.isfinite(q).all()
    # base was spun 3 rad/s * 2.4 s = 7.2 rad; the distal tip must pick up a large
    # fraction of that twist (transmission), with some lag (whip)
    assert tip_roll > 3.0
