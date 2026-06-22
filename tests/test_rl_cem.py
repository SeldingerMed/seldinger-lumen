"""Close the learning loop (doc M5): CEM policy search over the batched Layer-0 sim
actually learns — success rate improves and the trained policy navigates to target."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.assets import procedural
from lumen.envs import NavEnv
from lumen.rl.cem import make_policy, train_cem


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
    done = False
    info = {}
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs))
        done = term or trunc
    assert info["success"]
