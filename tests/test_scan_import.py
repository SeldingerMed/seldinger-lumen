"""Real-scan import on masks built from exact formulas (no external data).

Each mask is a tube rasterised on a scan grid with a known affine, so the true centreline and
radius are known. The gates are half a voxel of the scan's coarsest spacing.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from lumen.assets import scan_import as sc  # noqa: E402
from lumen.assets.schema import Asset  # noqa: E402


def _affine(spacing, origin, flip=(1, 1, 1)):
    A = np.eye(4)
    A[:3, :3] = np.diag(np.asarray(spacing, float) * np.asarray(flip, float))
    A[:3, 3] = origin
    return A


def _world_grid(shape, A):
    i, j, k = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    ijk = np.stack([i, j, k, np.ones_like(i)], axis=-1).astype(float)
    return ijk @ A.T


def _dist_to_polyline(P, poly):
    """Distance from each point of P (..., 3) to a polyline (n, 3)."""
    best = np.full(P.shape[:-1], np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(P - (a + t[..., None] * ab), axis=-1))
    return best


def _tube_mask(shape, A, polys, radii):
    W = _world_grid(shape, A)[..., :3]
    m = np.zeros(shape, bool)
    for poly, r in zip(polys, radii):
        m |= _dist_to_polyline(W, np.asarray(poly, float)) <= r
    return m


def _edge_points(asset):
    return np.concatenate([np.asarray(e.centerline_mm) for e in asset.edges])


def test_straight_tube_centreline_radius_and_position():
    spacing = (0.8, 0.8, 1.6)
    A = _affine(spacing, origin=(-40.0, 120.0, -900.0))
    poly = np.array([[-20.0, 140.0, -890.0], [-20.0, 140.0, -800.0]])
    mask = _tube_mask((50, 50, 70), A, [poly], [6.0])
    asset, report, tris = sc.asset_from_vessel_mask(sc.LabelVolume(mask.astype(np.uint8), A, {}))
    assert asset.provenance == "imported(scan)"
    assert asset.frame.name == "patient_lps_mm"
    assert len(asset.edges) == 1
    P = _edge_points(asset)
    half = 0.5 * max(spacing)
    mid = P[(P[:, 2] > -880) & (P[:, 2] < -810)]            # away from the flat end caps
    assert len(mid) > 50
    assert np.abs(mid[:, :2] - poly[0, :2]).max() < half      # absolute position in patient mm
    e = asset.edges[0]
    s = np.asarray(e.s_grid)
    R = np.asarray(e.R)[(s > 12) & (s < s[-1] - 12)]
    assert abs(np.median(R) - 6.0) < half
    assert np.abs(R - 6.0).max() < 2 * half
    # entry defaults to the caudal (low z) end
    assert report["entry_mm"][2] < -880
    # the surface is closed with outward normals: its volume matches the tube
    assert report["surface"]["surface_volume_mm3"] == pytest.approx(np.pi * 36 * 90, rel=0.08)
    assert Asset.from_dict(asset.to_dict()).edges[0].id == e.id


def test_flipped_axes_give_the_same_patient_geometry():
    """A scan stored right-to-left / top-down must land on the same patient millimetres."""
    poly = np.array([[10.0, 20.0, 5.0], [10.0, 20.0, 65.0]])
    shape = (40, 40, 60)
    A = _affine((1.0, 1.0, 1.25), origin=(-10.0, 0.0, 0.0))
    B = _affine((1.0, 1.0, 1.25), origin=(29.0, 0.0, 73.75), flip=(-1, 1, -1))
    out = []
    for aff in (A, B):
        mask = _tube_mask(shape, aff, [poly], [5.0])
        asset, _, tris = sc.asset_from_vessel_mask(sc.LabelVolume(mask.astype(np.uint8), aff, {}))
        P = _edge_points(asset)
        out.append(P[(P[:, 2] > 15) & (P[:, 2] < 55)])
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        assert np.einsum("ij,ij->i", tris.mean(1) - tris.mean(1).mean(0), n).sum() > 0   # outward
    for P in out:
        assert np.abs(P[:, :2] - poly[0, :2]).max() < 0.625


def test_y_fork_finds_both_branches_and_drops_a_loose_piece():
    spacing = (1.0, 1.0, 1.0)
    A = _affine(spacing, origin=(0.0, 0.0, 0.0))
    trunk = np.array([[40.0, 30.0, 4.0], [40.0, 30.0, 50.0]])
    left = np.array([[40.0, 30.0, 50.0], [20.0, 30.0, 90.0]])
    right = np.array([[40.0, 30.0, 50.0], [60.0, 30.0, 90.0]])
    mask = _tube_mask((80, 60, 96), A, [trunk, left, right], [7.0, 4.0, 4.0])
    mask[2:6, 2:6, 2:6] = True                                # a loose blob, not connected
    asset, report, _ = sc.asset_from_vessel_mask(sc.LabelVolume(mask.astype(np.uint8), A, {}))
    assert report["surface"]["pieces"] == 2
    assert report["surface"]["dropped_volume_mm3"] > 0
    assert asset.device_spawn.node_id == "inlet"
    ends = np.array([n.position_mm for n in asset.nodes if n.id.startswith("end")])
    assert len(ends) == 2
    for tip in (left[1], right[1]):
        assert np.linalg.norm(ends - tip, axis=1).min() < 6.0
    radii = sorted(e["median_radius_mm"] for e in report["edges"])
    assert abs(radii[0] - 4.0) < 0.6 and abs(radii[1] - 4.0) < 0.6 and abs(radii[-1] - 7.0) < 0.6
    assert report["wall_error_mm"]["source_to_lumen_outside_junction_zones_mm"]["p95"] < 0.5


def test_tilted_scan_is_refused():
    A = _affine((1.0, 1.0, 1.0), origin=(0, 0, 0))
    A[1, 2] = 0.3                                             # gantry tilt: sheared axes
    mask = np.zeros((20, 20, 20), np.uint8)
    mask[8:12, 8:12, 2:18] = 1
    with pytest.raises(ValueError, match="not perpendicular"):
        sc.asset_from_vessel_mask(sc.LabelVolume(mask, A, {}))


def test_empty_mask_is_refused():
    with pytest.raises(ValueError, match="mask is empty"):
        sc.asset_from_vessel_mask(sc.LabelVolume(np.zeros((8, 8, 8), np.uint8), np.eye(4), {}))


def test_smooth_keeps_short_polylines_finite():
    """A two- or three-point edge must not be thrown far away by the smoothing kernel."""
    from lumen.assets import surface_import as si

    for n in (2, 3, 4, 6):
        poly = np.stack([np.zeros(n), np.zeros(n), np.linspace(0.0, 0.7, n)], axis=1)
        out = si.smooth(poly, 4.0)
        assert out.shape == poly.shape
        assert np.abs(out - poly).max() < 0.7


def test_a_small_gap_is_joined_and_a_large_one_is_not():
    """A vessel broken by a 1 mm gap is one vessel; a piece 6 mm away is not attached."""
    shape, A = (40, 40, 110), _affine((0.5, 0.5, 0.5), (0.0, 0.0, 0.0))
    W = _world_grid(shape, A)[..., :3]
    z = W[..., 2]
    tube = _dist_to_polyline(W, np.array([[10.0, 10, 0], [10.0, 10, 60]])) <= 2.0
    # flat-ended pieces: 3..25 mm, then two empty slices, 26.5..40 mm, then a 6 mm gap, 46..52 mm
    mask = tube & (((z >= 3.0) & (z <= 25.0)) | ((z >= 26.5) & (z <= 40.0)) | ((z >= 46.0) & (z <= 52.0)))
    tris, info = sc.surface_from_mask(mask, A)
    assert len(info["joined"]) == 1 and 0.5 <= info["joined"][0]["gap_mm"] <= 2.0
    assert 24.0 <= info["joined"][0]["at_mm"][2] <= 27.5
    zs = tris[..., 2]
    assert zs.min() < 4.0 and 39.0 < zs.max() < 42.0             # both halves, not the far piece
    assert info["dropped_volume_mm3"] > 20.0
    tris0, info0 = sc.surface_from_mask(mask, A, join_mm=0.0)
    assert info0["joined"] == [] and tris0[..., 2].max() < 27.0


def test_nrrd_without_three_spatial_axes_is_refused_clearly(tmp_path):
    nrrd = pytest.importorskip("nrrd")
    data = np.zeros((4, 4, 4), np.uint8)
    hdr = {"space": "left-posterior-superior", "space origin": [0.0, 0.0, 0.0],
           "space directions": [[1.0, 0, 0], [0, 1.0, 0], [np.nan, np.nan, np.nan]]}
    f = tmp_path / "m.nrrd"
    nrrd.write(str(f), data, hdr, index_order="F")
    with pytest.raises(ValueError, match="three spatial axes"):
        sc.load_label_volume(f)
