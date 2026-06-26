"""M5 gymnasium integration: the lumen nav envs register and run through gymnasium.make.

Skips without gymnasium (it's an optional extra) or without warp/newton (the env steps
the Newton sim). The scene factories are also checked directly — those need no gymnasium."""

import numpy as np
import pytest

from lumen.envs.registration import LUMEN_ENVS, register_gym_envs


def test_scene_factories_build_steppable_envs():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    for factory in LUMEN_ENVS.values():           # raw factories: no gymnasium needed
        env = factory(max_steps=3)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (5,) and np.isfinite(obs).all()
        _, r, term, trunc, info = env.step(np.array([1.0], np.float32))
        assert "success" in info and np.isfinite(r)


def test_register_and_make_through_gymnasium():
    pytest.importorskip("gymnasium")
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    import gymnasium

    ids = register_gym_envs()
    assert set(ids) == set(LUMEN_ENVS)
    assert register_gym_envs() == ids             # idempotent
    assert all(i in gymnasium.registry for i in ids)

    env = gymnasium.make("Lumen/NavTube-v0")       # the standard external-consumption path
    obs, info = env.reset(seed=0)
    assert obs.shape == (5,)
    obs, r, term, trunc, info = env.step(np.array([1.0], np.float32))
    assert np.isfinite(r) and "success" in info
    assert env.unwrapped.R == 2.0                  # the underlying lumen env is reachable
