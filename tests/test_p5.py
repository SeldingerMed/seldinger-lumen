"""P5 / doc M5: Gym navigation env over the Newton Layer-0 solver."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.envs import NavEnv
from benchmarks.leaderboard import proportional_policy


def test_nav_env_reset_step_contract():
    env = NavEnv(target_frac=0.7, max_steps=6, device="cpu")
    obs, info = env.reset(seed=0)
    assert obs.shape == (5,) and np.isfinite(obs).all()
    a = proportional_policy(obs)
    assert a.shape == (2,) and -1.0 <= float(a[0]) <= 1.0 and a[1] == pytest.approx(0.0)
    if hasattr(env, "action_space"):
        assert env.action_space.shape == (2,)
    done, steps, reward = False, 0, 0.0
    while not done and steps < 6:
        obs, reward, term, trunc, info = env.step(a)
        a = proportional_policy(obs)
        done = term or trunc
        steps += 1
    assert np.isfinite(obs).all()
    assert isinstance(reward, float)
    assert set(info) >= {"tip_s", "dist", "max_r", "success"}
    assert np.isfinite(env.sim.body_positions()).all()


def test_nav_env_twist_action_reaches_distal_tip():
    env = NavEnv(target_frac=0.9, max_steps=8, max_twist=0.6, device="cpu")
    env.reset(seed=0)
    roll0 = env._tip_roll()
    info = {}
    for _ in range(6):
        obs, reward, term, trunc, info = env.step([0.0, 1.0])
        assert np.isfinite(obs).all()
        if term or trunc:
            break
    assert info["twist"] == pytest.approx(1.0)
    assert np.isfinite(info["tip_roll"])
    assert abs(info["tip_roll"] - roll0) > 0.05


def test_nav_env_success_boundary_is_inclusive():
    class Sim:
        def step(self, **_):
            pass

    env = object.__new__(NavEnv)
    env.sim = Sim()
    env.substeps = 1
    env.max_insertion = 1.0
    env.max_twist = 1.0
    env.steps = 0
    env.target_s = 10.0
    env.R = 2.0
    env._prev_dist = 5.0
    env.success_tol = 2.5
    env.max_steps = 5
    env._tip = lambda: (7.5, 0.0, 0.0, 2.0)
    env._tip_roll = lambda: 0.0
    env._obs = lambda: np.zeros(5, dtype=np.float32)

    _, _, terminated, _, info = env.step([0.0])

    assert terminated is True
    assert info["success"] is True
    assert info["dist"] == pytest.approx(2.5)


def test_nav_env_reports_unsafe_target_reach_as_unsafe_success():
    class Sim:
        def step(self, **_):
            pass

    env = object.__new__(NavEnv)
    env.sim = Sim()
    env.substeps = 1
    env.max_insertion = 1.0
    env.max_twist = 1.0
    env.steps = 0
    env.target_s = 10.0
    env.R = 2.0
    env._prev_dist = 2.0
    env.success_tol = 2.5
    env.max_steps = 5
    env.safety_max_pen = 0.3
    env._tip = lambda: (10.0, 3.0, 0.0, 3.0)
    env._contact_features = lambda: (3.0, 0.6)
    env._tip_roll = lambda: 0.0
    env._obs = lambda: np.zeros(5, dtype=np.float32)

    _, reward, terminated, truncated, info = env.step([0.0])

    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["safe_success"] is False
    assert info["unsafe"] is True
    assert info["max_pen"] == pytest.approx(0.6)
    assert reward < 10.0


def test_nav_env_penalizes_against_local_lumen_radius_not_mean_radius():
    class Projection:
        def __init__(self, s, theta, r):
            self.s = s
            self.theta = theta
            self.r = r

    class Frame:
        def project(self, p):
            return Projection(s=float(p[0]), theta=0.0, r=float(p[1]))

    class Lumen:
        def eval(self, s, theta=0.0):
            return 1.0 if s >= 9.0 else 2.0

    class Sim:
        def step(self, **_):
            pass

        def body_positions(self):
            return np.array([[0.0, 0.2, 0.0], [10.0, 1.6, 0.0]])

    env = object.__new__(NavEnv)
    env.sim = Sim()
    env.frame = Frame()
    env.lumen = Lumen()
    env.substeps = 1
    env.max_insertion = 1.0
    env.max_twist = 1.0
    env.steps = 0
    env.target_s = 12.0
    env.R = 2.0
    env._prev_dist = 4.0
    env.success_tol = 0.5
    env.max_steps = 5
    env._tip = lambda: (10.0, 1.6, 0.0, 1.6)
    env._tip_roll = lambda: 0.0
    env._obs = lambda: np.zeros(5, dtype=np.float32)
    env.safety_max_pen = 1.0

    _, reward, terminated, _, info = env.step([0.0])

    assert terminated is False
    assert info["max_pen"] == pytest.approx(0.6)
    assert reward == pytest.approx((4.0 - 2.0) - 0.5 * 0.6 - 0.01)
