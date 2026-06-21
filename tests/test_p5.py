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
