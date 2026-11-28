"""Mesh importer on synthetic procedural surfaces (no external data)."""

from __future__ import annotations

import json
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


def test_a_site_on_a_two_point_edge_splits_into_two_real_halves():
    edges = [("a", "b", np.array([[0.0, 0, 0], [0.4, 0, 0]]))]
    nodes = {"a": np.zeros(3), "b": np.array([0.4, 0, 0])}
    pi._split_at(edges, nodes, "site", np.array([0.1, 0, 0]))
    assert [len(p) for _, _, p in edges] == [2, 2]
    assert np.allclose(nodes["site"], [0.2, 0, 0])


def test_cli_writes_its_report_without_matplotlib(tmp_path: Path, monkeypatch):
    T = _tube([0, 0, 0], [0, 40, 0], 3.0, n_theta=64)
    f = tmp_path / "tube.stl"
    pi.write_stl(f, T)

    def no_matplotlib(*a, **k):
        raise ImportError("No module named 'matplotlib'")

    monkeypatch.setattr(pi, "_plots", no_matplotlib)
    pi.main(["--mesh", str(f), "--out", str(tmp_path / "out"), "--scale-to-mm", "1", "--mode", "solid",
             "--voxel-mm", "0.4", "--root", "0,1,0"])
    rep = json.loads((tmp_path / "out" / "import_report.json").read_text())
    assert rep["figures"] == [] and "conversion_error" in rep


def test_wall_fit_skips_only_points_past_the_entry_end():
    # the kernel holds the wall past every open end except the one the wire enters through
    import numpy as np

    from lumen.assets import procedural
    from lumen.assets.surface_import import contact_project
    from lumen.core import VascularTree

    tree = VascularTree(procedural.straight_tube(40.0, 2.0, n=20))       # wire enters at z = 0
    out = contact_project(tree, np.array([[1.0, 0.0, -3.0], [1.0, 0.0, 43.0], [1.0, 0.0, 20.0]]))
    assert out["culled"].tolist() == [True, False, False]


def test_despike_replaces_a_blip_along_the_vessel_and_keeps_the_rest_as_measured():
    rng = np.random.default_rng(0)
    R = 5.0 + 0.05 * rng.standard_normal((40, 32))
    R[20, 7:9] = 3.9                                # a 1.1 mm notch half a millimetre long
    R[11, 30] = 9.0                                 # a ray that ran into a neighbour
    out, bad = pi.despike(R)
    assert bad[20, 7] and bad[20, 8] and bad[11, 30] and bad.sum() == 3
    assert np.all(np.abs(out[bad] - 5.0) < 0.1)
    np.testing.assert_array_equal(out[~bad], R[~bad])


def test_despike_keeps_a_stenosis_a_bulge_and_a_branch_opening_with_their_corners():
    R = np.full((60, 32), 4.0)
    R[20:24, :] = 1.5                               # a 2 mm long, 63% stenosis (0.5 mm sections)
    R[40:46, 5:9] = 9.0                             # a branch opening 3 mm long, four rays wide
    R[50:53, 12:15] = 6.0                           # a bulge 1.5 mm long, three rays wide
    out, bad = pi.despike(R)
    assert not bad.any()
    np.testing.assert_array_equal(out, R)


def test_import_sizes_the_despike_window_in_millimetres(monkeypatch):
    """half_s counts sections; import passes 1 mm / step_mm, so the window stays 2 mm long and
    a 1.5 mm feature is kept at a finer step too."""
    seen, real = [], pi.despike

    def spy(R, **kw):
        seen.append(kw.get("half_s"))
        return real(R, **kw)

    monkeypatch.setattr(pi, "despike", spy)
    T = _tube([0, 0, 0], [0, 40, 0], 3.0, n_theta=64)
    side = pi.lumen_side(T, "solid")
    for step, half in ((0.5, 2), (0.25, 4)):
        seen.clear()
        pi.import_triangles(T, side, voxel_mm=0.4, root_mm=[0, 1.0, 0], step_mm=step)
        assert seen and set(seen) == {half}
    R = np.full((60, 32), 4.0)
    R[20:26, :] = 1.5                               # 1.5 mm long at a 0.25 mm step
    assert not real(R, half_s=4)[1].any()


def test_despike_keeps_the_deepest_point_of_a_tapering_stenosis():
    """A narrowing that tapers to its minimum over three sections (5, 3.5, 2, 3.5, 5 mm) keeps
    its 2 mm minimum: the sections either side lean the same way, so it is not a blip."""
    R = np.tile(np.array([5, 5, 5, 3.5, 2.0, 3.5, 5, 5, 5])[:, None], (1, 16))
    out, bad = pi.despike(R)
    assert not bad.any() and out.min() == 2.0
    W = R.copy()
    W[4] = 8.0                                      # a pocket at the same place is not tapered
    assert pi.despike(W)[1][4].all()


def test_the_shipped_cases_carry_the_despiked_wall():
    """Every catalogued case carries the despiked wall: before despiking 0.3 to 0.9% of the
    shipped radii were blips more than 0.3 mm from the median of their ray over 2 mm (a notch
    of 1.1 mm on the right subclavian of head-neck-cta, where the wire was judged through the
    wall). Despiked, a second pass finds almost none."""
    from lumen import cases

    names = cases.names()
    assert names
    for name in names:
        asset = cases.load(name)[0]
        n = total = 0
        for e in asset.edges:
            R = np.asarray(e.R, float)
            n += int(pi.despike(R)[1].sum())
            total += R.size
        assert n / total < 0.002, (name, n / total)
