"""Tube-intrinsic projection checks: the P0 deliverable.

A synthetic asset -> (s, theta, r) projection, verified against geometry we can
work out by hand.
"""

import numpy as np

from lumen.assets import procedural
from lumen.core.frame import CenterlineFrame


def test_straight_tube_projection():
    # centerline along +z; query points with known (s, theta, r)
    pts = np.stack([np.zeros(11), np.zeros(11), np.linspace(0, 100, 11)], axis=1)
    f = CenterlineFrame(pts)
    assert abs(f.length - 100.0) < 1e-9

    # offset +x at height 30 -> s=30, r=2, theta=0 (m1 seeds to +x)
    p = f.project(np.array([2.0, 0.0, 30.0]))
    assert abs(p.s - 30.0) < 1e-6
    assert abs(p.r - 2.0) < 1e-6
    assert abs(p.theta - 0.0) < 1e-6
    assert np.allclose(p.e_r, [1.0, 0.0, 0.0], atol=1e-6)

    # offset +y -> theta = +pi/2
    p = f.project(np.array([0.0, 1.5, 50.0]))
    assert abs(p.theta - np.pi / 2) < 1e-6
    assert abs(p.r - 1.5) < 1e-6


def test_project_s_batch_matches_scalar_project():
    # the vectorized batch projection must give the same arc-length as the per-point
    # project() (used on the batched flow-drag hot path)
    a = np.linspace(0, np.pi / 2, 40)
    cl = np.stack([30 * np.sin(a), 4 * np.cos(2 * a), 30 * (1 - np.cos(a))], axis=1)
    f = CenterlineFrame(cl)
    rng = np.random.default_rng(1)
    pts = cl[rng.integers(0, 40, 200)] + rng.normal(0, 1.0, (200, 3))
    s_loop = np.array([f.project(p).s for p in pts])
    s_vec = f.project_s(pts)
    assert np.allclose(s_loop, s_vec, atol=1e-9)
    assert f.project_s(pts[0]).shape == (1,)        # accepts a single point too


def test_curved_centerline_arclength_monotone():
    # quarter circle in the x-z plane, radius 20
    a = np.linspace(0, np.pi / 2, 50)
    pts = np.stack([20 * np.sin(a), np.zeros_like(a), 20 * (1 - np.cos(a))], axis=1)
    f = CenterlineFrame(pts)
    # arc length of a quarter circle r=20 is 20*pi/2 ~= 31.4
    assert abs(f.length - 20 * np.pi / 2) < 0.1
    # a point near the start projects to small s, near the end to large s
    p_start = f.project(pts[2] + np.array([0, 0.5, 0]))
    p_end = f.project(pts[-3] + np.array([0, 0.5, 0]))
    assert p_start.s < p_end.s


def test_gap_sign_via_lumen_field():
    asset = procedural.straight_tube(length=100, radius=2.0)
    pts, lf = asset.edge_arrays(asset.edges[0])
    f = CenterlineFrame(pts)
    # inside the lumen -> positive gap; outside -> negative
    p_in = f.project(np.array([1.0, 0.0, 50.0]))
    p_out = f.project(np.array([3.0, 0.0, 50.0]))
    assert lf.gap(p_in.s, p_in.theta, p_in.r) > 0
    assert lf.gap(p_out.s, p_out.theta, p_out.r) < 0
