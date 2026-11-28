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
    # 3 mm into the branch only the branch holds the point (just past the apex it is still
    # inside the wider trunk too, and the roomier lumen owns a point inside both)
    after = sharp.project(apex + 3.0 * _branch_dir()).R
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


def test_straight_tube_normalizes_a_valid_axis():
    asset = procedural.straight_tube(length=10.0, axis=(0.0, 3.0, 4.0))

    assert asset.nodes[-1].position_mm == pytest.approx((0.0, 6.0, 8.0))


@pytest.mark.parametrize(
    "axis",
    [
        (0.0, 0.0, 0.0),
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        (np.nan, 0.0, 1.0),
        (np.inf, 0.0, 1.0),
        ("bad", 0.0, 1.0),
        (10**400, 0.0, 1.0),
    ],
)
def test_straight_tube_rejects_invalid_axes(axis):
    with pytest.raises(ValueError, match="axis must be a finite nonzero 3-vector"):
        procedural.straight_tube(axis=axis)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("trunk", 0.0, "trunk must be a positive finite number"),
        ("trunk", np.inf, "trunk must be a positive finite number"),
        ("branch", -1.0, "branch must be a positive finite number"),
        ("branch", np.nan, "branch must be a positive finite number"),
        ("radius", 0.0, "radius must be a positive finite number"),
        ("radius", True, "radius must be a positive finite number"),
        ("angle_deg", np.inf, "angle_deg must be a finite number"),
        ("angle_deg", True, "angle_deg must be a finite number"),
        ("n", 7, "n must be an integer >= 8"),
        ("n", 8.5, "n must be an integer >= 8"),
        ("n", True, "n must be an integer >= 8"),
    ],
)
def test_bifurcation_rejects_invalid_parameters(parameter, value, message):
    with pytest.raises(ValueError, match=message):
        procedural.bifurcation(**{parameter: value})


def test_bifurcation_accepts_numpy_integer_sample_count():
    asset = procedural.bifurcation(n=np.int64(12))

    assert all(len(edge.centerline_mm) == 12 for edge in asset.edges)


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
        procedural.tortuous_tree(n=7)
    with pytest.raises(ValueError, match="n must be >= 8"):
        procedural.tortuous_tube(n=7)
    with pytest.raises(ValueError, match="dilation must be in"):
        procedural.tortuous_tube(dilation=1.1)
    with pytest.raises(ValueError, match="side_dilation must be in"):
        procedural.tortuous_tree(side_dilation=1.1)
    with pytest.raises(ValueError, match="severity must be in"):
        procedural.tortuous_tube(severity=-0.1)
    with pytest.raises(ValueError, match="severity must be in"):
        procedural.tortuous_tube(severity=0.9)
    with pytest.raises(ValueError, match="stenosis_severity must be in"):
        procedural.tortuous_tree(stenosis_severity=-0.1)
    with pytest.raises(ValueError, match="stenosis_severity must be in"):
        procedural.tortuous_tree(stenosis_severity=0.9)


def test_empty_asset_rejected():
    asset = procedural.straight_tube(10.0, 1.0)
    asset.edges = []
    with pytest.raises(ValueError, match="no edges"):
        VascularTree(asset)


def test_negative_blend_len_rejected():
    asset = procedural.straight_tube(10.0, 1.0)
    with pytest.raises(ValueError, match="blend_len must be >= 0"):
        VascularTree(asset, blend_len=-1.0)


@pytest.mark.parametrize("blend_len", [np.nan, np.inf])
def test_non_finite_blend_len_rejected(blend_len):
    asset = procedural.straight_tube(10.0, 1.0)
    with pytest.raises(ValueError, match="blend_len must be >= 0 and finite"):
        VascularTree(asset, blend_len=blend_len)


def test_generated_anatomy_keeps_its_tapered_outlets():
    # Only an end whose radius really closes (a scan's rounded cap) is held open; generated
    # outlets taper a little and stay as built.
    for build in (procedural.aortic_arch_tree, procedural.tortuous_tree, procedural.bifurcation):
        assert all(not caps for caps in VascularTree(build())._caps), build.__name__


def test_a_scan_cap_is_held_open():
    from lumen import cases

    held = sum(len(c) for cid in cases.names() for c in VascularTree(cases.load(cid)[0])._caps)
    assert held > 0


def test_a_diverged_point_projects_to_nans_instead_of_crashing():
    from lumen.assets import procedural
    from lumen.core import VascularTree

    pr = VascularTree(procedural.aortic_arch_tree()).project(np.array([np.nan, 0.0, 0.0]))
    assert not np.isfinite(pr.gap)


def _contact_owner(tree, q):
    """The edge the Newton contact kernel's rule gives a point (surface_import.contact_project,
    an independent NumPy copy of tube_barrier_kernel), and its room there."""
    from lumen.assets.surface_import import contact_project

    c = contact_project(tree, np.atleast_2d(np.asarray(q, float)))
    return int(c["edge"][0]), float(c["R"][0] - c["r"][0])


@pytest.mark.parametrize("name", ["arch", "bifurcation"])
def test_projection_and_the_contact_kernel_pick_the_same_vessel_near_junctions(name):
    """Issue #153: past a vessel's end, project() measured across the axis carried on beyond
    it while the kernel measured to the end point, so near junctions they disagreed on which
    vessel held a wire node (42 of 100 points on the arch agreed)."""
    from lumen.viewer.scene import resolve

    tree = VascularTree(resolve(name).asset)
    rng = np.random.default_rng(0)
    checked = 0
    for n in [n for n in tree._node_pos if tree.is_junction(n)]:
        for q in tree._node_pos[n] + rng.normal(size=(60, 3)) * 5.0:
            owner, room = _contact_owner(tree, q)
            if room > 0.1:                        # inside the lumen by the kernel's reading
                checked += 1
                assert tree.project(q).edge_index == owner
    assert checked >= 5


def test_at_a_bend_the_owner_is_chosen_by_the_distance_to_the_axis_not_across_it():
    """Two vessels of one radius leave a node; A bends sharply. Across A's axis the point is
    0.800 mm away, but its nearest point on A is the bend 1.0 mm away; B's end is 0.802 mm away.
    The kernel gives the point to B, and so must project()."""
    from lumen.assets.schema import Asset, DeviceSpawn, Frame, Node
    from lumen.core.lumen_field import LumenField

    def edge(eid, b, pts):
        pts = np.asarray(pts, float)
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        return procedural._edge_from_polyline(eid, "o", b, pts,
                                              LumenField.cylinder(length, 0.9, n=len(pts)))

    A = [(0.0, -0.75, 0.0), (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
    B = [(0.0, -0.75, 0.0), (0.0, 0.0, 0.05), (0.0, 0.2, 0.05)]
    asset = Asset(frame=Frame(), nodes=[Node("o", A[0]), Node("a", A[-1]), Node("b", B[-1])],
                  edges=[edge("A", "a", A), edge("B", "b", B)], device_spawn=DeviceSpawn(node_id="o"))
    tree = VascularTree(asset)
    q = np.array([0.0, 1.0, 0.0])
    owner, _ = _contact_owner(tree, q)
    assert tree.edges[owner].id == "B"
    assert tree.project(q).edge_id == "B"



def _elbow_tree():
    """A 2.0 mm vessel P runs up z into junction j, where a 2.6 mm vessel C turns off along x
    (no branch blending, so each vessel keeps its own radius)."""
    from lumen.assets.schema import Asset, DeviceSpawn, Frame, Node
    from lumen.core.lumen_field import LumenField

    def edge(eid, a, b, pts, R):
        pts = np.asarray(pts, float)
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        return procedural._edge_from_polyline(eid, a, b, pts,
                                              LumenField.cylinder(length, R, n=len(pts)))

    zs = np.linspace(-20.0, 0.0, 41)
    P = [(0.0, 0.0, z) for z in zs]
    C = [(x, 0.0, 0.0) for x in np.linspace(0.0, 20.0, 41)]
    asset = Asset(frame=Frame(), nodes=[Node("o", P[0]), Node("j", P[-1]), Node("c", C[-1])],
                  edges=[edge("P", "o", "j", P, 2.0), edge("C", "j", "c", C, 2.6)],
                  device_spawn=DeviceSpawn(node_id="o"))
    return VascularTree(asset, blend_len=0.0)


def test_past_a_junction_end_the_room_is_measured_from_the_end_point():
    """A point behind C's start, outside P's wall, belongs to C by the distance to C's axis
    (its end point). Its room there is C's radius minus that distance, not minus the distance
    across C's axis: measured across, a point 0.24 mm outside both vessels and the junction
    read 2.1 mm of room. project() and the contact kernel's rule agree."""
    tree = _elbow_tree()
    for q, room in (((-2.8, 0.0, -0.5), 2.6 - np.hypot(2.8, 0.5)),     # in the wall
                    ((-2.3, 0.0, -0.5), 2.6 - np.hypot(2.3, 0.5))):    # in the junction
        pr = tree.project(np.array(q))
        assert pr.edge_id == "C"
        assert abs(pr.gap - room) < 1e-6
        owner, croom = _contact_owner(tree, q)
        assert tree.edges[owner].id == "C" and abs(croom - room) < 1e-6
    assert tree.project(np.array((-2.8, 0.0, -0.5))).gap < -0.2
    # away from the ends nothing changes: across the axis
    assert abs(tree.project(np.array((0.0, 1.0, -5.0))).gap - 1.0) < 1e-6
    assert abs(tree.project(np.array((10.0, 0.0, 1.5))).gap - 1.1) < 1e-6


def test_the_wall_sampled_on_a_grid_equals_the_wall_point_by_point():
    """blended_R_grid (what the contact baker samples) is blended_R on every pair, caps and
    junction bulges included, on every shipped case and the synthetic trees."""
    from lumen import cases
    from lumen.viewer.scene import resolve

    rng = np.random.default_rng(1)
    assets = [cases.load(c)[0] for c in cases.names()] + [resolve(n).asset for n in ("arch", "bifurcation")]
    for asset in assets:
        tree = VascularTree(asset)
        for i, e in enumerate(tree.edges):
            ss = np.concatenate([[0.0, e.frame.length], rng.uniform(0, e.frame.length, 6),
                                 rng.uniform(0, min(6.0, e.frame.length), 3)])
            th = rng.uniform(-np.pi, np.pi, 4)
            grid = tree.blended_R_grid(i, ss, th)
            point = np.array([[tree.blended_R(i, float(a), float(b)) for b in th] for a in ss])
            np.testing.assert_allclose(grid, point, atol=1e-9)
