"""P5 / doc M4-M5: Gym nav env, leaderboard, throughput + bake-off harness."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lumen.envs import NavEnv
from benchmarks.leaderboard import run_leaderboard, proportional_policy
from benchmarks.throughput import run_throughput
from benchmarks.bakeoff import bench_primitive, _vbd_rod


def test_nav_env_reset_step_contract():
    env = NavEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (5,)
    obs, reward, term, trunc, info = env.step([0.5])
    assert obs.shape == (5,)
    assert isinstance(reward, float)
    assert set(info) >= {"tip_s", "dist", "max_r", "success"}


def test_proportional_policy_reaches_target():
    env = NavEnv(target_frac=0.7, max_steps=40)
    obs, _ = env.reset(seed=0)
    done = False
    info = {}
    while not done:
        obs, _, term, trunc, info = env.step(proportional_policy(obs))
        done = term or trunc
    assert info["success"]


def test_leaderboard_runs_and_succeeds():
    lb = run_leaderboard()
    assert lb["n_cases"] == 3
    assert lb["success_rate"] == 1.0
    assert all("peak_contact_r" in c for c in lb["cases"])


def test_tube_intrinsic_narrowphase_flat_in_T_and_wins_at_high_T():
    r32 = run_throughput(T=32, iters=20)
    r128 = run_throughput(T=128, iters=20)
    # tube-intrinsic cost is ~independent of circumferential resolution T...
    assert r128["t_tube_ms"] < 1.6 * r32["t_tube_ms"]
    # ...while the generic mesh cost grows, so the speedup increases and wins
    assert r128["speedup"] > r32["speedup"]
    assert r128["speedup"] > 1.0


def test_bakeoff_reports_throughput():
    r = bench_primitive(_vbd_rod, batch=16, steps=20)
    assert r["env_steps_per_s"] > 0
