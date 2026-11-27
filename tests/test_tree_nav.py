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


def test_a_reading_far_ahead_is_not_believed_however_long_the_tip_waits_there():
    env = _stub_env(30.0, prev=40.0)
    env._route_s = lambda p, near=None: 70.0     # read at the target, 40 mm on, every step
    env.target_s, env.max_steps = 70.0, 10
    for _ in range(6):
        _, _, terminated, _, info = env.step([0.0, 0.0])
        assert not info["success"] and not terminated
        assert info["tip_mm"] == pytest.approx(30.0) and info["off_route"]


def test_a_reading_far_ahead_is_believed_when_the_wire_lies_along_the_way():
    env = _stub_env(30.0, prev=40.0)
    # the tip is read at 70 mm and the wire behind it at 2 mm steps back to the last place
    places = iter([70.0] + [70.0 - 2.0 * k for k in range(1, 30)])
    env._route_s = lambda p, near=None: next(places)

    class Sim:
        def step(self, **_):
            pass

        def body_positions(self):
            return np.zeros((25, 3))

    env.sim = Sim()
    env.target_s = 70.0
    _, _, _, _, info = env.step([0.0, 0.0])
    assert info["tip_mm"] == pytest.approx(70.0) and not info["off_route"]


def test_a_wall_breach_put_back_by_containment_is_still_reported():
    env = _stub_env(5.0, prev=6.0)
    env.feed_stop_pen_mm, env.terminate_on_unsafe = 0.2, True
    env.L, env.success_tol, env._support_gap = 100.0, 2.5, 20.0
    env._contain = lambda: 0.6                   # a node 0.6 mm through, then moved back
    env._set_rail_along_wire = env._record_base_path = lambda: None
    env._solver_substeps = lambda dt: 1
    env.route_frame = type("F", (), {"project": staticmethod(
        lambda p: type("P", (), {"s": 0.0})())})()
    _, _, terminated, _, info = env.step([0.0, 0.0])
    assert info["max_pen"] == pytest.approx(0.6)
    assert info["unsafe"] and terminated


def test_a_point_past_the_end_of_the_route_is_not_in_it():
    env = TreeNavEnv(procedural.bifurcation(), target_node="apex", device="cpu")
    trunk = env.tree.edges[env.route[-1]].frame
    end, t = np.asarray(trunk.points[-1]), np.asarray(trunk.tangents[-1])
    assert env._route_s(end - 1.0 * t) == pytest.approx(env.L - 1.0, abs=0.2)
    assert env._route_s(end + 8.0 * t) is None


def test_a_short_route_gets_a_wire_that_fits_it():
    env = TreeNavEnv(procedural.straight_tube(24.0, 2.0), device="cpu")
    assert (env.wire_nodes - 1) * 2.0 <= env.L
    with pytest.raises(ValueError, match="does not fit"):
        TreeNavEnv(procedural.straight_tube(24.0, 2.0), device="cpu", wire_nodes=16)


@pytest.mark.parametrize("kw", [{"tip_deg": float("nan")}, {"contain_tol_mm": -1.0},
                                {"settle_steps": -1}, {"quasi_static": "false"}])
def test_bad_wire_settings_are_refused(kw):
    with pytest.raises((ValueError, TypeError)):
        TreeNavEnv(procedural.bifurcation(), target_node="left_out", device="cpu", **kw)


def test_a_benchmark_tip_off_the_route_is_not_a_clinical_target_reach():
    from types import SimpleNamespace

    from lumen.data.metrics import _tip_target

    def ep(on_route):
        step = SimpleNamespace(kinematics={"tip_s": 70.0, "on_route": on_route})
        return SimpleNamespace(steps=[step], outcome=SimpleNamespace(final_dist=0.0, success=False))
    notes = {"target_s": 70.0, "success_tol": 2.5}
    assert _tip_target(ep(1.0), notes)["success"] is True
    assert _tip_target(ep(0.0), notes)["success"] is False


def test_a_diverged_step_is_off_the_route_and_never_a_reach():
    from types import SimpleNamespace

    from lumen.data.metrics import _tip_target

    env = _stub_env(10.0, prev=10.0)                   # at the target, then the state blows up
    env._features = lambda: {"s": float("nan"), "r": 0.0, "theta": 0.0, "R_loc": 2.0,
                             "max_r": 2.0, "max_pen": 0.0, "edge": "left"}
    env._step_pen = 0.0
    _, _, terminated, _, info = env.step([0.0, 0.0])
    assert terminated and info["diverged"] and info["off_route"] and not info["success"]
    step = SimpleNamespace(kinematics={"tip_s": info["tip_mm"], "on_route": 0.0})
    ep = SimpleNamespace(steps=[step], outcome=SimpleNamespace(final_dist=0.0, success=False))
    assert _tip_target(ep, {"target_s": 10.0, "success_tol": 2.5})["success"] is False


def test_containment_works_before_the_solver_keeps_a_previous_state():
    import types

    pytest.importorskip("warp")
    pytest.importorskip("newton")
    env = TreeNavEnv(procedural.bifurcation(), target_node="left_out", device="cpu",
                     contain_tol_mm=0.0)
    sim = env.sim
    pos = sim.state_0.body_q.numpy().copy()             # on the CPU numpy() is a view
    b = sim.bodies[-1]
    pos[b, 0] += 50.0                                   # far through the wall
    sim.state_0.body_q.assign(pos)
    real = sim.solver
    sim.solver = types.SimpleNamespace(body_q_prev=None)
    try:
        pen = env._contain()
    finally:
        sim.solver = real
    assert pen > 10.0
    moved = np.linalg.norm(sim.state_0.body_q.numpy()[b, :3] - pos[b, :3])
    assert moved == pytest.approx(pen, rel=0.05)        # put back by the breach depth



@pytest.mark.parametrize("contain", [0.3, None])
def test_a_blown_up_wire_ends_the_episode_as_diverged(contain):
    """The simulator's own positions turn NaN (no stubbed projection): the step ends as
    diverged and off the route, with or without containment, instead of raising."""
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    env = TreeNavEnv(procedural.bifurcation(), target_node="left_out", device="cpu",
                     contain_tol_mm=contain)
    real = env.sim.body_positions
    env.sim.body_positions = lambda: np.full_like(np.asarray(real(), float), np.nan)
    _, _, terminated, _, info = env.step([1.0, 0.0])
    assert terminated and info["diverged"] and info["off_route"] and not info["success"]
