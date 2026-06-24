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


def test_empty_asset_rejected():
    asset = procedural.straight_tube(10.0, 1.0)
    asset.edges = []
    with pytest.raises(ValueError, match="no edges"):
        VascularTree(asset)


def test_negative_blend_len_rejected():
    asset = procedural.straight_tube(10.0, 1.0)
    with pytest.raises(ValueError, match="blend_len must be >= 0"):
        VascularTree(asset, blend_len=-1.0)
