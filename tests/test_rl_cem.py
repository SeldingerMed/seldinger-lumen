"""Close the learning loop (doc M5): CEM policy search over the batched Layer-0 sim
actually learns — success rate improves and the trained policy navigates to target."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.assets import procedural
from lumen.envs import NavEnv
from lumen.rl.cem import BatchedNav, make_policy, train_cem


def _straight():
    a = procedural.straight_tube(80.0, 2.0)
    pts, lumen = a.edge_arrays(a.edges[0])
    return np.asarray(pts), float(np.asarray(lumen.R).mean()), lumen


def test_cem_learns_to_navigate():
    vessel, R, lumen = _straight()
    # small/fast run for CI; learning still shows clearly on the straight task
    theta, hist = train_cem(vessel, R, lumen_field=lumen, pop=24, iters=8,
                            device="cpu", seed=0)
    assert hist[-1]["success_rate"] > hist[0]["success_rate"]    # it improved
    assert hist[-1]["success_rate"] >= 0.8                       # and converged high

    # the trained policy actually reaches the target on a real NavEnv episode
    env = NavEnv(asset=procedural.straight_tube(80.0, 2.0), target_frac=0.7, max_steps=40)
    obs, _ = env.reset(seed=0)
    policy = make_policy(theta)
    first_action = policy(obs)
    assert first_action.shape == (2,) and first_action[1] == pytest.approx(0.0)
    done = False
    info = {}
    while not done:
        obs, _, term, trunc, info = env.step(first_action)
        first_action = policy(obs)
        done = term or trunc
    assert info["success"]


def test_batched_rollout_success_boundary_is_inclusive():
    class Sim:
        def __init__(self):
            self.steps = 0

        def reset(self):
            self.steps = 0

        def step(self, **_):
            self.steps += 1

    env = object.__new__(BatchedNav)
    env.K = 1
    env.max_insertion = 1.0
    env.substeps = 1
    env.R = 2.0
    env.sim = Sim()
    env.target_s = 10.0
    calls = {"n": 0}

    def tip_obs():
        calls["n"] += 1
        s = np.array([0.0 if calls["n"] == 1 else 7.5])
        return np.zeros((1, 5), np.float32), s, np.zeros(1)

    env._tip_obs = tip_obs

    _, succ, steps = env.rollout(np.zeros((1, 6), np.float32), max_steps=3, success_tol=2.5)

    assert succ.tolist() == [True]
    assert steps.tolist() == [1.0]
    assert env.sim.steps == 1
