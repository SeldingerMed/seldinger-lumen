import sys
import types

import numpy as np
import pytest

from lumen.rl.adapters import make_cleanrl_env, make_cleanrl_vector_env, make_sb3_env


def test_cleanrl_thunk_seeds_spaces_and_records_episode_statistics():
    first = make_cleanrl_env("CartPole-v1", seed=11, idx=2)()
    second = make_cleanrl_env("CartPole-v1", seed=11, idx=2)()
    try:
        observation, info = first.reset(seed=13)
        assert observation.shape == (4,)
        assert info == {}
        np.testing.assert_array_equal(first.action_space.sample(), second.action_space.sample())
    finally:
        first.close()
        second.close()


def test_cleanrl_vector_adapter_builds_sync_vector_env():
    env = make_cleanrl_vector_env("CartPole-v1", num_envs=2, seed=7)
    try:
        observation, _ = env.reset(seed=21)
        assert observation.shape == (2, 4)
        actions = np.array([env.single_action_space.sample() for _ in range(2)])
        next_observation, _, terminated, truncated, _ = env.step(actions)
        assert next_observation.shape == (2, 4)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
    finally:
        env.close()


def test_cleanrl_vector_adapter_rejects_invalid_worker_count():
    with pytest.raises(ValueError, match="positive integer"):
        make_cleanrl_vector_env("CartPole-v1", num_envs=0)


def test_sb3_adapter_wraps_registered_env_with_monitor(monkeypatch):
    monitor_module = types.ModuleType("stable_baselines3.common.monitor")

    class FakeMonitor:
        def __init__(self, env):
            self.env = env

        def __getattr__(self, name):
            return getattr(self.env, name)

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

    monitor_module.Monitor = FakeMonitor
    common_module = types.ModuleType("stable_baselines3.common")
    package_module = types.ModuleType("stable_baselines3")
    monkeypatch.setitem(sys.modules, "stable_baselines3", package_module)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common_module)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.monitor", monitor_module)

    env = make_sb3_env("CartPole-v1", seed=5)
    try:
        assert isinstance(env, FakeMonitor)
        observation, _ = env.reset()
        assert observation.shape == (4,)
    finally:
        env.close()


def test_cleanrl_vector_adapter_runs_a_lumen_step():
    pytest.importorskip("warp")
    pytest.importorskip("newton")

    env = make_cleanrl_vector_env("Lumen/NavTube-v0", num_envs=1, seed=17)
    try:
        observation, _ = env.reset(seed=17)
        assert observation.shape == (1, 5)
        next_observation, reward, terminated, truncated, info = env.step(
            np.array([[1.0, 0.0]], dtype=np.float32)
        )
        assert next_observation.shape == (1, 5)
        assert np.isfinite(next_observation).all()
        assert np.isfinite(reward).all()
        assert terminated.shape == (1,)
        assert truncated.shape == (1,)
        assert "success" in info
    finally:
        env.close()


def test_sb3_adapter_runs_a_lumen_step_when_sb3_is_installed():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    pytest.importorskip("stable_baselines3")

    env = make_sb3_env("Lumen/NavTube-v0", seed=17, max_steps=1)
    try:
        observation, _ = env.reset(seed=17)
        assert observation.shape == (5,)
        next_observation, reward, terminated, truncated, info = env.step(
            np.array([1.0, 0.0], dtype=np.float32)
        )
        assert next_observation.shape == (5,)
        assert np.isfinite(next_observation).all()
        assert np.isfinite(reward)
        assert terminated or truncated
        assert "success" in info
    finally:
        env.close()


def test_cleanrl_lumen_video_request_fails_closed():
    pytest.importorskip("warp")
    pytest.importorskip("newton")

    with pytest.raises(ValueError, match="headless"):
        make_cleanrl_env("Lumen/NavTube-v0", capture_video=True)()
