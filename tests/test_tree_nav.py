"""L0d.1c — tree navigation: route helper (numpy) + TreeNavEnv (needs Newton)."""

import numpy as np
import pytest

from lumen.assets import procedural
from lumen.core import VascularTree
from lumen.envs.tree_nav import _HAS_GYM, TreeNavEnv, _route_polyline


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


def _stub_env(tip_s, prev, max_pen=0.0):
    """A TreeNavEnv with no solver: the tip is read inside the route at ``tip_s``."""

    class Sim:
        def step(self, **_):
            pass

        def body_positions(self):
            return np.zeros((3, 3))

    env = object.__new__(TreeNavEnv)
    env.sim = Sim()
    env.substeps, env.substep_ms = 1, 5.0
    env.max_insertion = env.max_twist = 1.0
    env.feed_stop_pen_mm = None
    env.quasi_static, env._feed, env._branch_mm = False, 0.0, ()
    env.steps, env.max_steps = 0, 5
    env.target_s, env.success_tol, env.safety_max_pen = 10.0, 2.5, 0.3
    env._prev = prev
    env._tip_s, env._tip_s_age, env._tip_best, env._progress_cap = tip_s, 0, tip_s, 10.0
    env._route_s = lambda p, near=None: tip_s
    env._features = lambda: {"s": tip_s, "r": 0.0, "theta": 0.0, "R_loc": 2.0,
                             "max_r": 2.0 + max_pen, "max_pen": max_pen, "edge": "left"}
    env._obs = lambda _: np.zeros(5, dtype=np.float32)
    env._tip_roll = lambda: 0.0
    return env


def test_tree_nav_success_boundary_is_inclusive_when_on_route():
    env = _stub_env(7.5, prev=5.0)

    _, _, terminated, _, info = env.step([0.0, 0.0])

    assert terminated is True
    assert info["success"] is True
    assert info["dist"] == pytest.approx(2.5)
    assert info["max_pen"] == pytest.approx(0.0)


def test_tree_nav_reports_unsafe_target_reach_as_unsafe_success():
    env = _stub_env(10.0, prev=2.0, max_pen=0.6)

    _, reward, terminated, truncated, info = env.step([0.0, 0.0])

    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["safe_success"] is False
    assert info["unsafe"] is True
    assert info["max_pen"] == pytest.approx(0.6)
    assert reward < 10.0


def test_a_tip_outside_the_route_vessels_is_off_route_and_not_paid():
    """Near the target's centreline but not inside a route vessel (the wrong branch, or across
    a coil): no success, no progress reward."""
    env = _stub_env(9.0, prev=5.0)
    env._route_s = lambda p, near=None: None
    env._tip_s = 5.0

    _, reward, terminated, _, info = env.step([0.0, 0.0])

    assert info["off_route"] is True
    assert info["success"] is False and terminated is False
    assert reward < 0.0


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
        assert env.action_space.shape == (2,) and env.observation_space.shape == (5,)
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
