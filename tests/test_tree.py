"""L0d.1a — vascular tree: graph projection + branch-node R blending (pure numpy)."""

import numpy as np
import pytest

from lumen.assets import procedural
from lumen.core import VascularTree


def _apex(asset):
    return np.asarray([n.position_mm for n in asset.nodes if n.id == "apex"][0], float)


def _branch_dir(angle_deg=35.0, sign=-1.0):
    a = np.radians(angle_deg)
    return np.array([sign * np.sin(a), 0.0, np.cos(a)])


def test_projection_picks_the_owning_edge():
    asset = procedural.bifurcation(angle_deg=35.0)
    tree = VascularTree(asset)
    apex = _apex(asset)
    assert tree.project(apex - np.array([0, 0, 10.0])).edge_id == "trunk"
    assert tree.project(apex + 12.0 * _branch_dir(sign=-1)).edge_id == "left"
    assert tree.project(apex + 12.0 * _branch_dir(sign=+1)).edge_id == "right"


def test_R_is_continuous_across_the_junction():
    asset = procedural.bifurcation(radius=2.0, angle_deg=35.0)   # trunk R=2.0, branch R=1.6
    tree = VascularTree(asset, blend_len=4.0)
    apex = _apex(asset)
    before = tree.project(apex - np.array([0, 0, 0.5])).R         # trunk side of the apex
    after = tree.project(apex + 0.5 * _branch_dir()).R           # branch side of the apex
    assert abs(before - after) < 0.2                             # no 2.0->1.6 step at the node
    assert before > 1.7                                          # bulged toward the trunk radius


def test_blending_relaxes_to_each_edges_own_radius_far_from_node():
    asset = procedural.bifurcation(radius=2.0, angle_deg=35.0)
    tree = VascularTree(asset, blend_len=4.0)
    apex = _apex(asset)
    far_trunk = tree.project(apex - np.array([0, 0, 25.0])).R
    far_branch = tree.project(apex + 30.0 * _branch_dir()).R
    assert abs(far_trunk - 2.0) < 0.05                          # trunk's own radius
    assert abs(far_branch - 1.6) < 0.05                         # branch's own radius


def test_blend_len_zero_gives_sharp_junction():
    asset = procedural.bifurcation(radius=2.0, angle_deg=35.0)
    sharp = VascularTree(asset, blend_len=0.0)
    apex = _apex(asset)
    after = sharp.project(apex + 0.5 * _branch_dir()).R
    assert abs(after - 1.6) < 0.05                              # no blend: steps straight to branch R


def test_gap_sign_and_junction_flag():
    asset = procedural.bifurcation(radius=2.0)
    tree = VascularTree(asset)
    apex = _apex(asset)
    inside = apex - np.array([0.0, 0.0, 20.0])                  # on the trunk centerline -> r~0
    assert tree.gap(inside) > 0                                 # clearance inside the lumen
    deep = inside + np.array([5.0, 0.0, 0.0])                   # 5mm off a ~2mm-radius tube
    assert tree.gap(deep) < 0                                   # outside the wall -> penetration
    assert tree.is_junction("apex") and not tree.is_junction("trunk_in")


def test_straight_tube_is_a_degenerate_tree():
    tree = VascularTree(procedural.straight_tube(80.0, 2.0))    # one edge, no junctions
    pr = tree.project(np.array([0.0, 0.0, 40.0]))
    assert pr.edge_id == "e0" and abs(pr.R - 2.0) < 1e-6 and pr.gap > 0


def test_project_edge_s_batches_flow_geometry_without_per_node_project_loop():
    asset = procedural.bifurcation(angle_deg=35.0)
    tree = VascularTree(asset)
    apex = _apex(asset)
    points = np.stack([
        apex - np.array([0.0, 0.0, 10.0]),
        apex + 12.0 * _branch_dir(sign=-1),
        apex + 12.0 * _branch_dir(sign=+1),
    ])

    edge, s = tree.project_edge_s(points)

    expected_edge = np.array([tree.project(p).edge_index for p in points])
    expected_s = np.array([tree.project(p).s for p in points])
    assert edge.shape == (3,)
    assert s.shape == (3,)
    assert np.array_equal(edge, expected_edge)
    assert np.allclose(s, expected_s)


def test_tortuous_tree_has_asymmetric_curved_tapered_vessels():
    asset = procedural.tortuous_tree(radius=2.3, n=32, stenosis_severity=0.45)
    tree = VascularTree(asset)

    assert asset.provenance == "procedural"
    assert asset.device_spawn.node_id == "inlet"
    assert len(asset.edges) == 5
    assert tree.is_junction("side_junction")
    assert tree.is_junction("apex")

    right = next(edge for edge in asset.edges if edge.id == "right_stenotic")
    right_pts = np.asarray(right.centerline_mm, float)
    right_r = np.asarray(right.R, float)[:, 0]
    assert np.ptp(right_pts[:, 0]) > 20.0
    assert np.ptp(right_pts[:, 2]) > 20.0
    assert right_r.min() < 0.75 * right_r[0]       # focal narrowing, not just taper
    assert right_r[-1] < right_r[0]                # distal taper
    side = next(edge for edge in asset.edges if edge.id == "side")
    side_r = np.asarray(side.R, float)[:, 0]
    side_peak = int(np.argmax(side_r))
    assert 4 < side_peak < len(side_r) - 5
    assert side_r[side_peak] > 1.08 * max(side_r[0], side_r[-1])

    route = tree.route("right_out", "inlet")
    assert [tree.edges[i].id for i in route] == ["trunk_in", "trunk_mid", "right_stenotic"]
    assert tree.route_length(route) > 120.0


def test_aortic_arch_tree_has_supra_aortic_branch_complexity():
    asset = procedural.aortic_arch_tree()
    tree = VascularTree(asset)

    assert len(asset.edges) >= 6
    assert asset.device_spawn.node_id == "inlet"
    assert {node.id for node in asset.nodes} >= {
        "inlet",
        "descending_out",
        "brachio_out",
        "carotid_out",
        "subclavian_out",
    }
    assert tree.is_junction("arch_prox")
    assert tree.is_junction("arch_mid")
    assert tree.route_length(tree.route("descending_out", "inlet")) > 120.0


def test_tortuous_tube_has_curvature_taper_and_focal_narrowing():
    asset = procedural.tortuous_tube(length=90.0, radius=3.0, severity=0.25, n=48)
    pts, lumen = asset.edge_arrays(asset.edges[0])
    pts = np.asarray(pts, float)
    radii = np.asarray(lumen.R, float)[:, 0]

    assert len(asset.edges) == 1
    assert np.ptp(pts[:, 0]) > 8.0
    assert np.ptp(pts[:, 1]) > 1.0
    assert np.ptp(pts[:, 2]) > 85.0
    assert radii[-1] < radii[0]
    assert radii.min() < 0.85 * radii[0]
    dilation_peak = int(np.argmax(radii))
    assert 4 < dilation_peak < len(radii) - 8
    assert radii[dilation_peak] > 1.05 * max(radii[0], radii[-1])
    assert np.all(radii > 0.0)


def test_tortuous_demo_assets_reject_invalid_parameters():
    with pytest.raises(ValueError, match="radius must be positive"):
        procedural.tortuous_tree(radius=0.0)
    with pytest.raises(ValueError, match="n must be >= 8"):
        procedural.tortuous_tube(n=7.8)


def test_empty_asset_rejected():
    asset = procedural.straight_tube(10.0, 1.0)
    asset.edges = []
    with pytest.raises(ValueError, match="no edges"):
        VascularTree(asset)


def test_negative_blend_len_rejected():
    asset = procedural.straight_tube(10.0, 1.0)
    with pytest.raises(ValueError, match="blend_len must be >= 0"):
        VascularTree(asset, blend_len=-1.0)
