"""The wire on real-anatomy-like bends: a siphon, a junction with a side vessel opening into the
route, a guide catheter parked in the large vessels, and the microcatheter's support limit."""

from __future__ import annotations

import numpy as np
import pytest

from lumen.assets import procedural
from lumen.assets.schema import Asset, DeviceSpawn, Frame, Node
from lumen.core.lumen_field import LumenField
from lumen.envs.tree_nav import TreeNavEnv, guide_start
from lumen.viewer.scene import resolve


def _solver():
    pytest.importorskip("warp")
    pytest.importorskip("newton")


def _edge(eid, a, b, pts, radius):
    pts = np.asarray(pts, float)
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return procedural._edge_from_polyline(eid, a, b, pts, LumenField.cylinder(length, radius, n=len(pts)))


def _arc(start, heading, rc, turn_deg, sign, n=60):
    """Points of a circular arc in the x-z plane leaving ``start`` along ``heading``; sign +1
    turns counter-clockwise (seen from +y... toward -x from +z), -1 the other way."""
    t = np.asarray(heading, float) / np.linalg.norm(heading)
    left = sign * np.array([-t[2], 0.0, t[0]])           # the centre is on this side
    c = np.asarray(start, float) + rc * left
    u0 = (np.asarray(start, float) - c) / rc
    pts = []
    for a in np.radians(np.linspace(0.0, turn_deg, n))[1:]:
        # rotate u0 about y by -sign*a... use the in-plane basis (u0, t)
        pts.append(c + rc * (np.cos(a) * u0 + np.sin(a) * t))
    return pts


def siphon(radius=2.4, rc=9.0, lead=3.0, turn=120.0) -> Asset:
    """A carotid-siphon-like S: two opposite arcs of ``turn`` degrees and centreline radius
    ``rc`` (curvature radius well above the lumen radius), then straight."""
    pts = [np.array([0.0, 0.0, z]) for z in np.linspace(0.0, lead, max(2, int(lead) + 1))]
    for sign in (1, -1):
        t = pts[-1] - pts[-2]
        pts += _arc(pts[-1], t, rc, turn, sign)
    t = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
    pts += [pts[-1] + t * d for d in np.linspace(1.0, 40.0, 40)]
    pts = np.asarray(pts)
    return Asset(frame=Frame(), nodes=[Node("in", tuple(pts[0])), Node("out", tuple(pts[-1]))],
                 edges=[_edge("e0", "in", "out", pts, radius)], device_spawn=DeviceSpawn(node_id="in"))


def _joint_angles(p):
    d = np.diff(p, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1.0, 1.0)))


def test_a_wire_laid_through_a_siphon_holds_still_and_keeps_its_shape():
    """With 5 ms solver steps the rod did not converge in a tight bend: joints folded past
    140 degrees and the tip slid back with no input."""
    _solver()
    env = TreeNavEnv(siphon(), max_steps=10**6)
    env.reset()
    p0 = np.asarray(env.sim.body_positions(), float)
    for _ in range(15):
        env.step(np.zeros(2, np.float32))
    p = np.asarray(env.sim.body_positions(), float)
    assert np.linalg.norm(p[-1] - p0[-1]) < 1.0
    assert _joint_angles(p).max() < 60.0
    rest = np.linalg.norm(np.diff(p0, axis=0), axis=1).sum()
    assert abs(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() - rest) < 0.03 * rest


def test_pushing_through_a_siphon_moves_the_tip_with_the_hand():
    _solver()
    env = TreeNavEnv(siphon(), max_steps=10**6)
    env.reset()
    f = env.route_frame
    p0 = np.asarray(env.sim.body_positions(), float)
    worst = 0.0
    for _ in range(20):
        _, _, _, _, info = env.step(np.array([1.0, 0.0], np.float32))
        worst = max(worst, info["max_pen"])
    p = np.asarray(env.sim.body_positions(), float)
    base = f.project(p[0]).s - f.project(p0[0]).s
    tip = f.project(p[-1]).s - f.project(p0[-1]).s
    assert base > 10.0
    assert tip > 0.8 * base
    assert worst < 0.5
    assert info["furthest_mm"] >= f.project(p[-1]).s - 1e-6


def _route_with_wide_stub() -> Asset:
    """A 2 mm route vessel with a short, wider (3.5 mm) side vessel leaving it at 40 mm."""
    z = np.linspace(0.0, 40.0, 41)
    trunk = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    cont = trunk[-1] + np.stack([np.zeros(41), np.zeros(41), np.linspace(0.0, 40.0, 41)], axis=1)
    d = np.array([np.sin(np.radians(25)), 0.0, np.cos(np.radians(25))])
    stub = trunk[-1] + np.linspace(0.0, 10.0, 11)[:, None] * d
    return Asset(frame=Frame(),
                 nodes=[Node("in", tuple(trunk[0])), Node("j", tuple(trunk[-1])),
                        Node("out", tuple(cont[-1])), Node("stub_end", tuple(stub[-1]))],
                 edges=[_edge("trunk", "in", "j", trunk, 2.0), _edge("cont", "j", "out", cont, 2.0),
                        _edge("stub", "j", "stub_end", stub, 3.5)],
                 device_spawn=DeviceSpawn(node_id="in"))


def test_progress_counts_a_tip_in_the_route_vessel_where_a_wider_side_vessel_owns_it():
    """Tree ownership goes to the edge with the most room, so just past the junction a point on
    the route's own axis belongs to the wider side vessel. It is still in the route vessel."""
    env = TreeNavEnv(_route_with_wide_stub(), target_node="out")
    p = np.array([1.0, 0.0, 43.0])                              # 1 mm off the route's axis
    pr = env.tree.project(p)
    assert env.tree.edges[pr.edge_index].id == "stub"          # the case this guards against
    env._tip_best = env._tip_s = 30.0
    assert env._route_s(p) == pytest.approx(43.0, abs=0.2)
    assert env._route_s(np.array([6.0, 0.0, 45.0])) is None    # in the side vessel only


def test_a_guide_catheter_starts_the_wire_beyond_the_wide_vessels():
    _solver()
    asset = resolve("arch").asset
    env = TreeNavEnv(asset, target_node="carotid_out", guide_radius_mm=3.0)
    assert env.start_node != asset.device_spawn.node_id
    assert float(np.mean(env.tree.edges[env.route[0]].lf.R)) < 3.0
    full = TreeNavEnv(asset, target_node="carotid_out")
    assert env.L < full.L
    with pytest.raises(ValueError, match="catheter-only"):
        TreeNavEnv(asset, target_node="descending_out", guide_radius_mm=3.0)
    # a guide wider than every vessel on the route stays at the entry
    assert guide_start(full.tree, full.route, full.start_node, 100.0) == full.start_node


def test_the_pushed_end_stays_the_support_gap_behind_a_tip_that_cannot_go_on():
    """Pushing a wire whose tip is at the end of the vessel does not feed it into a coil: the
    microcatheter stops support_gap_mm behind the tip."""
    _solver()
    env = TreeNavEnv(procedural.straight_tube(60.0, 2.0), max_steps=10**6, support_gap_mm=20.0)
    env.reset()
    f = env.route_frame
    for _ in range(40):
        env.step(np.array([1.0, 0.0], np.float32))
    p = np.asarray(env.sim.body_positions(), float)
    assert f.project(p[0]).s <= f.project(p[-1]).s - 20.0 + 0.5


def test_bad_wire_parameters_are_refused():
    asset = procedural.straight_tube(60.0, 2.0)
    for kw in ({"substep_ms": 0.0}, {"bend_damping": -1.0}, {"support_gap_mm": float("nan")},
               {"wire_nodes": 3}):
        with pytest.raises(ValueError):
            TreeNavEnv(asset, **kw)


@pytest.mark.parametrize("tiny", [0.6, 0.005])
def test_a_very_short_route_vessel_still_tracks_the_tip(tiny):
    """The route keeps every vessel's own points, so a vessel far shorter than a wire segment
    (a junction split a hair apart by the skeleton) still has its stretch of the route."""
    z = np.linspace(0.0, 40.0, 41)
    a = np.c_[0 * z, 0 * z, z]
    c = np.c_[0 * z, 0 * z, 40.0 + tiny + z]
    asset = Asset(frame=Frame(),
                  nodes=[Node("in", (0.0, 0.0, 0.0)), Node("j1", (0.0, 0.0, 40.0)),
                         Node("j2", (0.0, 0.0, 40.0 + tiny)), Node("out", tuple(c[-1]))],
                  edges=[_edge("a", "in", "j1", a, 2.0),
                         _edge("t", "j1", "j2", [[0.0, 0.0, 40.0], [0.0, 0.0, 40.0 + tiny]], 2.0),
                         _edge("c", "j2", "out", c, 2.0)],
                  device_spawn=DeviceSpawn(node_id="in"))
    env = TreeNavEnv(asset, target_node="out", device="cpu")
    env._tip_best = env._tip_s = 40.0
    assert env._route_s(np.array([0.0, 0.0, 40.0 + tiny / 2])) == pytest.approx(40.0 + tiny / 2, abs=0.01)
    assert env._route_s(np.array([0.0, 0.0, 45.0])) == pytest.approx(45.0, abs=0.01)


def test_a_spawn_node_missing_from_the_asset_is_refused():
    from lumen.core.tree import VascularTree

    asset = siphon()
    asset.device_spawn = DeviceSpawn(node_id="nowhere")
    with pytest.raises(ValueError, match="device spawn node 'nowhere'"):
        VascularTree(asset)
