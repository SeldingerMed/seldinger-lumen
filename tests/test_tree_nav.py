"""L0d.1c — tree navigation: route helper (numpy) + TreeNavEnv (needs Newton)."""

import numpy as np
import pytest

from lumen.assets import procedural
from lumen.core import VascularTree
from lumen.envs.tree_nav import _HAS_GYM, _route_polyline


# ---- route graph helpers (pure numpy) ----------------------------------------
def test_route_finds_the_branch_path():
    asset = procedural.bifurcation(trunk=50.0, branch=50.0)
    tree = VascularTree(asset)
    route = tree.route("left_out", "trunk_in")             # trunk -> apex -> left
    assert [tree.edges[i].id for i in route] == ["trunk", "left"]
    assert abs(tree.route_length(route) - 100.0) < 1.0     # ~trunk + branch length


def test_route_polyline_is_continuous_across_the_junction():
    asset = procedural.bifurcation(trunk=50.0, branch=50.0)
    tree = VascularTree(asset)
    route = tree.route("right_out", "trunk_in")
    poly = _route_polyline(tree, route, "trunk_in")
    steps = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    assert steps.max() < 3.0 and steps.min() > 1e-6        # no gap/dup at the apex
    assert np.allclose(poly[0], [0, 0, 0], atol=1e-6)      # starts at the trunk inlet


def test_route_to_unknown_node_raises():
    tree = VascularTree(procedural.bifurcation())
    with pytest.raises(ValueError, match="unknown node"):
        tree.route("does_not_exist", "trunk_in")
    with pytest.raises(ValueError, match="unknown node"):     # false-empty-path: target==start, both unknown
        tree.route("ghost", "ghost")


def test_route_between_leaves_goes_through_apex():
    tree = VascularTree(procedural.bifurcation())
    route = tree.route("right_out", "left_out")            # left -> apex -> right
    assert [tree.edges[i].id for i in route] == ["left", "right"]


def test_empty_route_polyline_is_single_point():
    tree = VascularTree(procedural.bifurcation())
    poly = _route_polyline(tree, [], "trunk_in")           # start == target -> []
    assert poly.shape == (1, 3)


def test_env_rejects_target_equal_start():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.envs import TreeNavEnv
    with pytest.raises(ValueError, match="differ from start"):
        TreeNavEnv(procedural.bifurcation(), target_node="trunk_in", device="cpu")


# ---- the env (needs the Layer-0 tree sim) ------------------------------------
def test_tree_nav_env_builds_steps_and_progresses():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.envs import TreeNavEnv

    env = TreeNavEnv(procedural.bifurcation(trunk=50.0, branch=50.0, angle_deg=25.0),
                     target_node="left_out", max_steps=40, device="cpu")
    obs, _ = env.reset()
    assert obs.shape == (5,) and np.isfinite(obs).all()
    if _HAS_GYM:
        assert env.action_space.shape == (1,) and env.observation_space.shape == (5,)
    last = None
    for _ in range(35):
        obs, r, term, trunc, info = env.step(1.0)           # push forward along the route
        assert np.isfinite(obs).all()
        last = info
        if term or trunc:
            break
    # the wire advances PAST the trunk (50mm) into the branch — the route-following base
    # actuation fix (it would clamp at the apex if the base only followed the trunk).
    assert env._features()["s"] > 52.0                      # crossed the junction into the branch
    assert last["max_r"] <= env.R + 0.3 + 0.15             # held within the lumen band
    # honest: this pins the route pipeline + route-following actuation; it does NOT claim
    # learned branch SELECTION (the route is prescribed; the policy controls insertion).
