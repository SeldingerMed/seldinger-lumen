import numpy as np
import pytest

from lumen.assets import procedural


def _envs(n, **extra):
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.envs.tree_nav import TreeNavEnv
    from lumen.envs.tree_nav_batched import BatchedTreeNavEnv
    asset = procedural.bifurcation(trunk=50.0, branch=50.0, radius=2.0, angle_deg=45.0)
    kw = dict(target_node="left_out", max_insertion=1.8, max_steps=20, device="cpu", **extra)
    return TreeNavEnv(asset, **kw), BatchedTreeNavEnv(asset, n_envs=n, **kw)


def test_batched_worlds_match_single_env_and_reset_independently():
    single, batched = _envs(3)
    o1, _ = single.reset()
    ob, _ = batched.reset()
    start = batched.sim.env_positions().copy()
    assert ob.shape == (3, 5)
    np.testing.assert_allclose(ob, np.repeat(o1[None], 3, 0), atol=1e-6)
    rng = np.random.default_rng(0)
    acts = rng.uniform(-1, 1, (6, 3, 2)).astype(np.float32)
    acts[:, 0] = np.array([[1.0, 0.0]] * 6, np.float32)      # world 0 replays the single env's actions
    for t in range(6):
        o1, r1, *_ = single.step(acts[t, 0])
        ob, rb, term, trunc, infos = batched.step(acts[t])
    np.testing.assert_allclose(ob[0], o1, atol=1e-3)
    assert rb[0] == pytest.approx(r1, abs=1e-3)
    assert len(infos) == 3 and not trunc.any()
    before = batched.sim.env_positions().copy()
    init = batched.reset_env(1)
    after = batched.sim.env_positions()
    np.testing.assert_allclose(after[0], before[0])
    np.testing.assert_allclose(after[2], before[2])
    np.testing.assert_allclose(after[1], start[1], atol=1e-5)
    assert not np.allclose(before[1], start[1])                # world 1 had moved before its reset
    np.testing.assert_allclose(init, single.reset()[0], atol=1e-6)
    assert batched.steps[1] == 0 and batched.steps[0] == 6


def test_batched_rejects_bad_n_envs():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.envs.tree_nav_batched import BatchedTreeNavEnv
    with pytest.raises(ValueError):
        BatchedTreeNavEnv(procedural.bifurcation(trunk=50.0, branch=50.0, radius=2.0), n_envs=0)


def test_start_jitter_is_seeded_bounded_and_per_world():
    plain, _ = _envs(1)
    single, batched = _envs(3, start_jitter_mm=1.0)
    plain.reset()
    base = plain.sim.body_positions().copy()

    def offset(env):
        d = env.sim.body_positions() - base
        np.testing.assert_allclose(d, np.repeat(d[:1], len(d), 0), atol=1e-5)   # the whole device moves rigidly
        return d[0]

    single.reset(seed=7)
    a = offset(single)
    single.reset(seed=7)
    np.testing.assert_allclose(offset(single), a, atol=1e-6)                  # same seed, same start
    single.reset(seed=8)
    b = offset(single)
    assert not np.allclose(a, b) and np.linalg.norm(a) <= 1.0 + 1e-6 and np.linalg.norm(b) <= 1.0 + 1e-6
    single.reset()
    assert not np.allclose(offset(single), b)                                 # unseeded resets keep drawing
    batched.reset(seed=3)
    d = batched.sim.env_positions() - base[None]
    starts = d[:, 0]
    assert len({tuple(np.round(s, 6)) for s in starts}) == 3                  # each world draws its own offset
    assert (np.linalg.norm(starts, axis=1) <= 1.0 + 1e-6).all()
    old = batched.sim.env_positions().copy()
    batched.reset_env(1)
    new = batched.sim.env_positions()
    np.testing.assert_allclose(new[0], old[0])
    assert not np.allclose(new[1], old[1]) and np.linalg.norm(new[1][0] - base[0]) <= 1.0 + 1e-6
    with pytest.raises(ValueError):
        _envs(1, start_jitter_mm=-1.0)
