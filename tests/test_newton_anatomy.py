"""Regression tests for anatomy-correctness of the contact kernel (#1, #4, #5).

These exercise the geometry the old kernel silently broke: a non-cylindrical
R(s) (stenosis), and a curved, non-uniformly sampled, +x-aligned centerline
(which made the old hardcoded m1=(1,0,0) degenerate and the uniform-spacing s
wrong). All checks are at the kernel level and compare against lumen.core.frame.
"""

import numpy as np
import pytest

pytest.importorskip("warp")
import warp as wp

from lumen.core.frame import CenterlineFrame
from lumen.newton.tube_barrier_kernel import accumulate_tube_barrier


def _launch(cl, R0_grid, body_pos, n_s, n_th, d_hat=0.3, kappa=2e3):
    f = CenterlineFrame(cl)
    P = wp.array(f.points.astype(np.float32), dtype=wp.vec3)
    Tg = wp.array(f.tangents.astype(np.float32), dtype=wp.vec3)
    M1 = wp.array(f.m1.astype(np.float32), dtype=wp.vec3)
    cum_s = wp.array(f.cum_s.astype(np.float32), dtype=wp.float32)
    r0 = wp.array(np.asarray(R0_grid, np.float32).ravel(), dtype=wp.float32)
    bq = wp.array(np.array([[*body_pos, 0, 0, 0, 1]], np.float32), dtype=wp.transform)
    bqd = wp.array(np.zeros((1, 6), np.float32), dtype=wp.spatial_vector)
    cg = wp.array(np.array([0], np.int32), dtype=wp.int32)
    wm = wp.array(np.array([1], np.int32), dtype=wp.int32)
    wf = wp.zeros(n_s * n_th, dtype=wp.float32)
    ld = wp.zeros(n_s * n_th, dtype=wp.float32)
    bf = wp.zeros(1, dtype=wp.vec3); bh = wp.zeros(1, dtype=wp.mat33)
    wp.launch(accumulate_tube_barrier, dim=1,
              inputs=[cg, wm, bq, bqd, P, Tg, M1, cum_s, len(f.points), r0,
                      float(f.length), n_s, n_th, wf, kappa, d_hat, 0, 0.0, 0.0, 0.0],
              outputs=[bf, bh, ld])
    return f, np.array(bf.numpy()[0]), ld.numpy()


def test_stenosis_R_reaches_contact():
    # #1: R(s) (not a scalar mean) drives contact. Straight tube, R=2 except a
    # narrowing to R=1.3 near s=40. A node at r=1.6 contacts ONLY at the stenosis.
    M, n_s, n_th = 41, 41, 8
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, 80, M)], axis=1)
    s_grid = np.linspace(0, 80, n_s)
    R_s = np.where(np.abs(s_grid - 40) < 6, 1.3, 2.0)          # stenosis profile R(s)
    R0_grid = np.repeat(R_s[:, None], n_th, axis=1)
    _, f_sten, _ = _launch(cl, R0_grid, (1.6, 0, 40), n_s, n_th)   # at the stenosis
    _, f_open, _ = _launch(cl, R0_grid, (1.6, 0, 10), n_s, n_th)   # wide segment
    assert abs(f_sten[0]) > 100.0          # strong inward contact where R is small
    assert abs(f_open[0]) < 1e-3           # no contact where the lumen is wide
    # a scalar mean would have made these identical — proves R(s) is read


def test_curved_nonuniform_plusx_centerline_matches_frame():
    # #4 + #5: curved, NON-uniformly sampled, +x-aligned centerline. The kernel's
    # (s,θ) cell for a contact must match lumen.core.frame (which the old hardcoded
    # m1 and uniform-s kernel got wrong).
    a = np.concatenate([np.linspace(0, 0.3, 6), np.linspace(0.34, np.pi / 3, 16)])
    cl = np.stack([40 * np.sin(a), np.zeros_like(a), 40 * (1 - np.cos(a))], axis=1)
    n_s, n_th = 24, 12
    f = CenterlineFrame(cl)
    # query point: a mid-curve foot pushed outward in the +y direction
    j = len(cl) // 2
    p = cl[j] + 0.6 * np.array([0.0, 1.0, 0.0])
    pr = f.project(p)
    exp_is = int(np.clip(pr.s / f.length * (n_s - 1) + 0.5, 0, n_s - 1))
    th01 = (pr.theta + np.pi) / (2 * np.pi)
    exp_ith = int(th01 * n_th) % n_th
    exp_cell = exp_is * n_th + exp_ith
    # R0 small so the node penetrates and deposits load in its cell
    R0_grid = np.full(n_s * n_th, 0.5, np.float32)
    _, force, load = _launch(cl, R0_grid, tuple(p), n_s, n_th)
    assert abs(force).sum() > 1.0                       # contact happened (non-degenerate frame)
    assert load.argmax() == exp_cell                   # load landed in frame.py's (s,θ) cell
    assert load[exp_cell] > 0.0
