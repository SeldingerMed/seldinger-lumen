"""Batched throughput: the GPU-scaling property the fast tier must keep (§3.3).

We can't assert the bible's absolute 1e4 env-steps/s here (that's a CUDA figure;
CI is CPU). What we CAN pin, on any device, is the *scaling*: the per-env-step
cost must fall sharply as the batch grows. That only holds if the substep loop has
no per-env host round-trip — a stray .numpy() (device->host sync) inside the loop
would serialize the envs and flatten the curve. This test is that regression guard.
"""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.newton.flow import FlowField
from lumen.newton.sim import NewtonGuidewireSim
from lumen.newton.throughput import measure_throughput


def _scene(M=30, L=60.0, n=9):
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.zeros(n), np.zeros(n), np.linspace(4, 4 + 2 * (n - 1), n)], axis=1)
    return vessel, dev


def _us_per_env_step(E, coupled=False, steps=8):
    vessel, dev = _scene()
    flow = FlowField() if coupled else None       # flow -> n_envs>1 takes the _step_device path
    sim = NewtonGuidewireSim(vessel, 2.0, dev, radius=0.2, n_envs=E, flow=flow, device="cpu")
    return sim, measure_throughput(sim, steps=steps, substeps=3, insertion=1.0)["us_per_env_step"]


def test_measure_throughput_reports_sane_fields():
    vessel, dev = _scene()
    sim = NewtonGuidewireSim(vessel, 2.0, dev, radius=0.2, n_envs=4, device="cpu")
    r = measure_throughput(sim, steps=5, substeps=3, insertion=1.0)
    assert r["n_envs"] == 4 and r["device"] == "cpu"
    assert r["env_steps_per_s"] > 0 and r["ms_per_step"] > 0
    # env-steps/s == n_envs * steps / elapsed == n_envs / (ms_per_step/1e3) — internally consistent
    assert r["env_steps_per_s"] == pytest.approx(4 * 1e3 / r["ms_per_step"], rel=1e-6)


def test_measure_throughput_rejects_nonpositive_sizes():
    vessel, dev = _scene()
    sim = NewtonGuidewireSim(vessel, 2.0, dev, radius=0.2, n_envs=2, device="cpu")
    with pytest.raises(ValueError, match="steps must be positive"):
        measure_throughput(sim, steps=0)


def test_batching_amortizes_on_the_host_path():
    # plain navigation (no flow/clot) -> _step_host. Batching must drive the per-env
    # cost WELL below the single-env cost; a per-env host round-trip would flatten it.
    _, one = _us_per_env_step(1)
    _, many = _us_per_env_step(32)
    assert many < 0.5 * one        # >=2x cheaper per env at batch 32 — amortization holds


def test_batching_amortizes_on_the_device_coupled_path():
    # flow co-sim with n_envs>1 -> the on-device _step_device loop (the real GPU path).
    # Compare two batched sizes that BOTH take it; per-env cost must still fall sharply,
    # i.e. the per-substep co-sim has no per-env device->host sync.
    small_sim, small = _us_per_env_step(4, coupled=True)
    big_sim, big = _us_per_env_step(64, coupled=True)
    assert small_sim._use_device_coupling and big_sim._use_device_coupling   # the _step_device path
    assert big < 0.5 * small       # batched on-device co-sim amortizes too
