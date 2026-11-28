"""The wire stays in the vessel when pushed against a bend or a dead end."""

from __future__ import annotations

import numpy as np
import pytest

from lumen.envs.tree_nav import TreeNavEnv
from lumen.viewer.scene import resolve


def _solver():
    """Tests that step the wire need the optional solver; the rest run without it."""
    pytest.importorskip("warp")
    pytest.importorskip("newton")


def _push(env, steps):
    _solver()
    env.reset()
    worst, info = 0.0, {}
    for _ in range(steps):
        _, _, _, _, info = env.step(np.array([1.0, 0.0], np.float32))
        worst = max(worst, info["max_pen"])
    return worst, info


SHARP_SIDE = pytest.mark.xfail(strict=False, reason=(
    "side_out leaves a 7.6 mm trunk at 64 degrees: the tip enters only from a 5 mm approach "
    "at a third of rolls, then the wire hooks and more push prolapses it up the straight-ahead "
    "trunk. Getting in needs the microcatheter advanced into the branch, not modelled yet."))


def _follow(anatomy, target):
    """Drive the wire to ``target`` with the probing FollowPolicy for up to 400 steps; return
    the last step's info and the deepest wall penetration on the way."""
    from lumen.viewer.record import FollowPolicy

    _solver()
    # an arch case is worked from a guide catheter parked in the aorta, not a bare wire
    guide = 4.5 if anatomy == "arch" else None
    env = TreeNavEnv(resolve(anatomy).asset, target_node=target, max_steps=400,
                     guide_radius_mm=guide)
    env.reset()
    policy, info, worst = FollowPolicy(), {"dist": 1e9, "off_route": False}, 0.0
    for _ in range(400):
        _, _, term, _, info = env.step(np.asarray(policy(info), np.float32))
        worst = max(worst, info["max_pen"])
        if term:
            break
    return info, worst


@pytest.mark.parametrize("anatomy,target", [
    ("tortuous", "right_out"), pytest.param("tortuous", "side_out", marks=SHARP_SIDE),
    ("arch", "brachio_out"), ("arch", "carotid_out")])
def test_a_steered_wire_reaches_each_branch_without_going_through_the_wall(anatomy, target):
    info, worst = _follow(anatomy, target)
    assert info["success"]
    assert worst < 1.5


def test_a_wire_prolapsing_at_the_arch_left_subclavian_stays_in_the_vessel():
    """The arch's left subclavian and descending aorta leave one junction 39 degrees apart and
    share an 11 mm funnel. With the microcatheter held 20 mm behind the tip it stays in the
    funnel, and each push loops the bare wire down the descending aorta round the carina; the
    tip creeps about 1 mm a try. Whether it arrives within 400 steps turns on microns (walls
    2-10 um narrower reach, 5-10 um wider do not), so arrival is not asserted here: getting in
    reliably needs the microcatheter advanced into the branch, not modelled yet. However the
    wire loops, it stays in the vessel."""
    info, worst = _follow("arch", "subclavian_out")
    assert not info.get("diverged", False)     # a blown-up episode reports no penetration
    assert worst < 1.5
    assert info["furthest_mm"] > 50.0          # into the branch: past its carina at 49 mm


def test_a_wide_vessel_owns_a_point_near_a_thin_branch():
    """A point inside a wide lumen is inside the vessel even when a thin branch's axis is
    nearer: judging it against the thin wall would push the wire the wrong way."""
    from lumen.core.tree import VascularTree

    from lumen.assets.schema import Asset

    a = resolve("arch").asset.to_dict()
    # shrink the brachiocephalic branch to a 0.4 mm thread leaving the 13 mm-wide arch
    for e in a["edges"]:
        if e["id"] == "brachiocephalic":
            e["R"] = [[0.4] * len(row) for row in e["R"]]
    tree = VascularTree(Asset.from_dict(a), blend_len=0.0)
    thread = next(e for e in tree.edges if e.id == "brachiocephalic")
    # a point 3 mm up the thread and 1 mm beside its axis: the thread's axis is the nearest
    # one and the point is outside the thread's 0.4 mm wall, but well inside the arch
    i = int(np.searchsorted(thread.frame.cum_s, 3.0))
    p = thread.frame.points[i] + 1.0 * thread.frame.m1[i]
    near = min(tree.edges, key=lambda e: e.frame.project(p).r)
    assert near.id == "brachiocephalic"                  # the old nearest-axis owner
    got = tree.project(p)
    assert got.edge_id != "brachiocephalic" and got.gap > 1.0


def test_pushing_into_a_dead_end_loads_the_wire_instead_of_feeding_it_out():
    env = TreeNavEnv(resolve("tortuous").asset, target_node="right_out", max_steps=300)
    worst, _ = _push(env, 300)                       # far more push than the route can take
    assert worst < 1.2
    tip = env.route_frame.project(env.sim.body_positions()[-1]).s
    assert tip <= env.L + 1.0


def test_the_feed_limit_can_be_turned_off():
    env = TreeNavEnv(resolve("tortuous").asset, target_node="right_out", feed_stop_pen_mm=None)
    assert env.feed_stop_pen_mm is None
    _push(env, 3)


def test_density_must_be_a_positive_number():
    a = resolve("tortuous").asset
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            TreeNavEnv(a, target_node="right_out", density=bad)


# ------------------------------------------------------------------ imported cross-sections
def test_a_section_cut_along_a_neighbour_is_rebuilt_from_the_vessel_around_it():
    from lumen.assets.surface_import import along_vessel

    R = np.full((60, 32), 0.8)
    R[30:33] = 20.0                         # every ray long: the section's own median passes
    R2, odd = along_vessel(R)
    assert odd[30:33].all() and np.allclose(R2, 0.8)


def test_a_ray_that_escapes_along_the_wall_is_capped():
    from lumen.assets.surface_import import along_vessel

    R = np.full((60, 32), 0.5)
    R[40, :6] = [6.0, 17.0, 27.0, 24.0, 21.0, 18.0]   # centre line on the wall of a thin artery
    R2, _ = along_vessel(R)
    assert R2.max() <= 3.0 * 0.5 + 1.0 + 1e-9


@pytest.mark.parametrize("kind", ["aneurysm", "stenosis", "origin"])
def test_real_shape_changes_are_kept(kind):
    from lumen.assets.surface_import import along_vessel

    if kind == "aneurysm":                  # 2.8x bulge over 8 sections
        R = np.full((60, 32), 1.0)
        R[26:34] = 2.8
    elif kind == "stenosis":                # 80% narrowing in diameter
        R = np.full((60, 32), 2.0)
        R[28:33] = 0.4
    else:                                   # an artery widening smoothly into the aorta
        R = np.repeat(np.r_[np.linspace(16.0, 5.0, 40), np.full(60, 5.0)][:, None], 32, axis=1)
    R2, odd = along_vessel(R)
    assert not odd.any() and np.allclose(R2, R)


def test_a_vessel_cut_off_at_the_scan_edge_does_not_close_to_a_point():
    from lumen.assets.schema import Asset
    from lumen.core.tree import VascularTree

    a = resolve("tortuous").asset.to_dict()
    e = next(x for x in a["edges"] if x["node_b"] == "right_out")
    n = len(e["R"])
    base = float(e["R"][n // 2][0])
    for k in range(1, 6):                   # a rounded cap over the last few samples
        e["R"][-k] = [base * (k - 1) / 5.0 + 0.05 for _ in e["R"][-k]]
    tree = VascularTree(Asset.from_dict(a), blend_len=0.0)
    i = next(i for i, x in enumerate(tree.edges) if x.node_b == "right_out")
    end = tree.edges[i].frame.length
    assert tree.blended_R(i, end, 0.0) > 0.8 * base


def test_the_entry_end_is_open_and_holds_its_width():
    """The wire comes in through the entry: no end wall there, and the scan's closing cap at
    the cut does not squeeze the lumen (it is held at its width just inside, like every other
    cut end)."""
    import numpy as np

    from lumen import cases
    from lumen.core.tree import CAP_CLOSING, VascularTree, is_open_entry

    asset, _ = cases.load("head-cta-aneurysm")
    tree = VascularTree(asset)
    assert tree.entry_node is not None
    assert is_open_entry(tree, tree.entry_node)
    for i, e in enumerate(tree.edges):
        for end_s, node in ((0.0, e.node_a), (e.frame.length, e.node_b)):
            if node != tree.entry_node:
                continue
            inside = min(abs(e.frame.length - end_s), 1.5 * float(np.median(np.asarray(e.lf.R))))
            ref = end_s + (inside if end_s == 0.0 else -inside)
            th = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
            at_cut = np.mean([tree.blended_R(i, end_s, t) for t in th])
            just_in = np.mean([tree.blended_R(i, ref, t) for t in th])
            assert at_cut >= (1.0 - CAP_CLOSING) * just_in - 0.02
