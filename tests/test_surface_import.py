"""Mesh importer on synthetic procedural surfaces (no external data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from lumen.assets import surface_import as pi  # noqa: E402


def _tube(p0, p1, r, n_theta=48, n_len=40, cap0=True, cap1=True, inward=False):
    """Triangulated tube along p0->p1 (radius r); optional flat end caps. Outward normals
    unless ``inward``."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = (p1 - p0) / np.linalg.norm(p1 - p0)
    a = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(t, a)
    u /= np.linalg.norm(u)
    v = np.cross(t, u)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ring = lambda c: c + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)  # noqa: E731
    rings = [ring(p0 + (p1 - p0) * k / n_len) for k in range(n_len + 1)]
    tris = []
    for k in range(n_len):
        A, B = rings[k], rings[k + 1]
        for i in range(n_theta):
            j = (i + 1) % n_theta
            tris += [[A[i], A[j], B[j]], [A[i], B[j], B[i]]]
    if cap0:
        tris += [[p0, rings[0][(i + 1) % n_theta], rings[0][i]] for i in range(n_theta)]
    if cap1:
        tris += [[p1, rings[-1][i], rings[-1][(i + 1) % n_theta]] for i in range(n_theta)]
    T = np.asarray(tris)
    # orient outward: normal must point away from the axis
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    c = T.mean(1)
    rel = c - p0
    radial = rel - (rel @ t)[:, None] * t
    axial = (rel @ t) - 0.5 * np.linalg.norm(p1 - p0)
    out_dir = np.where(np.linalg.norm(radial, axis=1)[:, None] > 0.5 * r, radial, axial[:, None] * t)
    flip = np.einsum("ij,ij->i", nrm, out_dir) < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    if inward:
        T = T[:, [0, 2, 1]]
    return T


def test_stl_roundtrip(tmp_path: Path):
    T = _tube([0, 0, 0], [0, 0, 10], 2.0)
    f = tmp_path / "t.stl"
    pi.write_stl(f, T)
    assert np.allclose(pi.read_stl(f), T, atol=1e-5)


def test_straight_tube_solid(tmp_path: Path):
    T = _tube([0, 0, 0], [0, 40, 0], 3.0, n_theta=64)
    f = tmp_path / "tube.stl"
    pi.write_stl(f, T)
    asset, rep = pi.import_mesh(f, scale_to_mm=1.0, mode="solid", voxel_mm=0.4, root_mm=[0, 1.0, 0])
    assert rep["voxelization"]["inconsistent_crossings"] == 0
    assert abs(rep["voxelization"]["volume_mm3"] - np.pi * 9 * 40) / (np.pi * 9 * 40) < 0.05
    assert len(asset.edges) == 1 and rep["skeleton"]["leaves"] == 1
    R = np.asarray(asset.edges[0].R)
    mid = R[len(R) // 4: 3 * len(R) // 4]
    assert np.allclose(mid, 3.0, atol=0.1)
    err = pi.conversion_error(asset, f, scale_to_mm=1.0, mode="solid", n_samples=20_000)
    assert err["lumen_to_source_mm"]["p95"] < 0.3          # under one voxel (0.4 mm)


def test_shell_matches_solid():
    """A thin-walled open-ended shell (as the phantom is): outer tube, inward-facing inner tube and
    flat end rings. The inner surface is labelled lumen-facing and voxelizes to the inner lumen."""
    inner = _tube([0, 0, 0], [0, 30, 0], 2.5, cap0=False, cap1=False, inward=True)
    outer = _tube([0, 0, 0], [0, 30, 0], 2.8, cap0=False, cap1=False)
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ring = []
    for y in (0.0, 30.0):
        for i in range(48):
            j = (i + 1) % 48
            a_i = np.array([2.5 * np.cos(th[i]), y, 2.5 * np.sin(th[i])])
            a_j = np.array([2.5 * np.cos(th[j]), y, 2.5 * np.sin(th[j])])
            b_i = np.array([2.8 * np.cos(th[i]), y, 2.8 * np.sin(th[i])])
            b_j = np.array([2.8 * np.cos(th[j]), y, 2.8 * np.sin(th[j])])
            ring += [[a_i, b_j, b_i], [a_i, a_j, b_j]]
    T = np.concatenate([outer, inner, np.asarray(ring)])
    side = pi.lumen_side(T, "shell")
    assert (side[len(outer):len(outer) + len(inner)] == 1).all()
    assert (side[:len(outer)] == 0).all() and (side[len(outer) + len(inner):] == 0).all()
    g, vox = pi.voxelize(T, side, 0.3)
    assert vox["inconsistent_crossings"] == 0
    vol = np.pi * 2.5**2 * 30
    assert abs(vox["volume_mm3"] - vol) / vol < 0.06


def test_shell_mode_refuses_a_closed_surface():
    """A closed surface of the lumen is not an open shell; saying so beats a wrong lumen."""
    capped = _tube([0, 0, 0], [0, 40, 0], 3.0, n_theta=64)                 # barrel and two end caps
    with pytest.raises(ValueError, match="mode='solid'"):
        pi.lumen_side(capped, "shell")
    barrel = _tube([0, 0, 0], [0, 40, 0], 3.0, cap0=False, cap1=False)     # one smooth patch
    with pytest.raises(ValueError, match="mode='solid'"):
        pi.lumen_side(barrel, "shell")
    assert (pi.lumen_side(capped, "solid") == -1).all()


def test_y_bifurcation_branches_and_sites():
    parts = [_tube([0, 0, 0], [0, 30, 0], 4.0, n_theta=64), _tube([0, 26, 0], [-18, 55, 0], 2.0),
             _tube([0, 26, 0], [18, 55, 0], 2.5)]
    # overlapping closed tubes are not one clean surface: voxelize each on a shared grid and OR them.
    # A zero-side triangle spanning the union bounds fixes the grid without bounding any lumen.
    lo = np.min([q.reshape(-1, 3).min(0) for q in parts], axis=0) - 1
    hi = np.max([q.reshape(-1, 3).max(0) for q in parts], axis=0) + 1
    span = np.array([[lo, [hi[0], lo[1], lo[2]], hi]])
    occ, ref = None, None
    for q in parts:
        g, vox = pi.voxelize(np.concatenate([q, span]), np.r_[-np.ones(len(q)), 0.0], 0.4)
        assert vox["inconsistent_crossings"] == 0
        occ, ref = (g.occ.copy(), g) if occ is None else (occ | g.occ, ref)
    sk = pi.centerline_tree(pi.Grid(ref.origin, ref.h, occ), np.array([0, 1.0, 0]))
    assert sk.info["leaves"] == 2 and sk.info["junctions"] == 1
    ends = [sk.nodes[n] for n in sk.nodes if n.startswith("end")]
    assert sorted(np.sign([e[0] for e in ends]).tolist()) == [-1.0, 1.0]
    edges = list(sk.edges)
    nodes = dict(sk.nodes)
    d = pi._split_at(edges, nodes, "target", np.array([-12.0, 46.0, 0.0]))
    assert d < 1.5 and "target" in nodes and len(edges) == 4
