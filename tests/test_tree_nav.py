"""L0d.1c — tree navigation: route helper (numpy) + TreeNavEnv (needs Newton)."""

import numpy as np
import pytest

from lumen.assets import procedural
from lumen.core import VascularTree
from lumen.envs.tree_nav import _route_polyline


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
    with pytest.raises(ValueError, match="no route"):
        tree.route("does_not_exist", "trunk_in")


def test_route_between_leaves_goes_through_apex():
    tree = VascularTree(procedural.bifurcation())
    route = tree.route("right_out", "left_out")            # left -> apex -> right
    assert [tree.edges[i].id for i in route] == ["left", "right"]


# ---- the env (needs the Layer-0 tree sim) ------------------------------------
def test_tree_nav_env_builds_steps_and_progresses():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.envs import TreeNavEnv

    env = TreeNavEnv(procedural.bifurcation(trunk=50.0, branch=50.0, angle_deg=25.0),
                     target_node="left_out", max_steps=30, device="cpu")
    obs, _ = env.reset()
    assert obs.shape == (5,) and np.isfinite(obs).all()
    s0 = env._tip()[0]
    last = None
    for _ in range(20):
        obs, r, term, trunc, info = env.step(1.0)           # push forward
        assert np.isfinite(obs).all()
        last = info
        if term or trunc:
            break
    # the wire advances along the route (up the trunk at least) and stays in the lumen
    assert env._tip()[0] > s0 + 10.0                        # real forward route progress
    assert last["max_r"] <= env.R + 0.3 + 0.15             # held within the lumen band
    # honest: reaching the branch LEAF needs steering/branch-selection (not asserted) —
    # this pins the env + route pipeline, not learned vessel choice.
