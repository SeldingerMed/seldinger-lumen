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
    assert a.shape == (1,) and -1.0 <= float(a[0]) <= 1.0
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


def test_nav_env_success_boundary_is_inclusive():
    class Sim:
        def step(self, **_):
            pass

    env = object.__new__(NavEnv)
    env.sim = Sim()
    env.substeps = 1
    env.max_insertion = 1.0
    env.steps = 0
    env.target_s = 10.0
    env.R = 2.0
    env._prev_dist = 5.0
    env.success_tol = 2.5
    env.max_steps = 5
    env._tip = lambda: (7.5, 0.0, 0.0, 2.0)
    env._obs = lambda: np.zeros(5, dtype=np.float32)

    _, _, terminated, _, info = env.step([0.0])

    assert terminated is True
    assert info["success"] is True
    assert info["dist"] == pytest.approx(2.5)
