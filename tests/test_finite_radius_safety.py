"""Finite-radius contact (P0-0): the device SURFACE, not the centerline, is what
breaches the wall.

The tube/tree barriers and the NavEnv penetration proxy must all account for the
configured device radius. Geometry is deterministic: the expected surface
penetration of a node at radius r with device radius rho in a tube of radius R is
``max(0, r + rho - R)`` — one radius clear, one overlapping.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.assets import procedural                       # noqa: E402
from lumen.envs import NavEnv, TreeNavEnv                 # noqa: E402
from lumen.newton.sim import NewtonGuidewireSim           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TUBE_LEN = 60.0
TUBE_R = 2.0


def _straight_centerline():
    return np.stack([np.zeros(30), np.zeros(30), np.linspace(0.0, TUBE_LEN, 30)], axis=1)


def _device_at_offset(offset, n=10, sp=2.0):
    """Deterministic device centerline: straight along z, offset in +x."""
    return np.stack([
        np.full(n, offset),
        np.zeros(n),
        np.arange(n) * sp,
    ], axis=1)


def _make_sim(device_offset, device_radius, **kw):
    return NewtonGuidewireSim(
        vessel_centerline=_straight_centerline(),
        R=TUBE_R,
        device_points=_device_at_offset(device_offset),
        radius=device_radius,
        device="cpu",
        **kw,
    )


class TestSurfacePenetrationGeometry:
    """The penetration proxy must equal max(0, r + rho - R) node-wise."""

    @pytest.mark.parametrize("offset,rho,expected", [
        (1.0, 0.2, 0.0),
        (1.9, 0.2, 0.1),
        (1.0, 0.9, 0.0),
        (1.5, 0.9, 0.4),
        (1.5, 0.2, 0.0),
    ])
    def test_two_radii_same_centerline(self, offset, rho, expected):
        sim = _make_sim(offset, rho)
        pen = sim.surface_penetration()
        assert pen.shape == (len(sim.bodies),)
        assert np.allclose(pen, expected, atol=1e-6), (
            f"offset={offset} rho={rho}: expected {expected}, got {pen}"
        )

    def test_thick_overlaps_where_thin_is_clear(self):
        offset = 1.5
        pen_thin = _make_sim(offset, 0.2).surface_penetration()
        pen_thick = _make_sim(offset, 0.9).surface_penetration()
        assert pen_thin.max() == pytest.approx(0.0, abs=1e-6)
        assert pen_thick.max() == pytest.approx(0.4, abs=1e-6)


class TestBarrierWallLoad:
    """The wall barrier must feel a thick device before a thin one."""

    def test_tube_barrier_load_grows_with_radius(self):
        offset = 1.7
        loads = []
        for rho in (0.05, 0.6):
            sim = _make_sim(offset, rho)
            sim.step(dt=2.5e-2, substeps=2)
            loads.append(float(np.asarray(sim.wall_load_grid()).sum()))
        assert loads[1] > loads[0]

    def test_tree_barrier_load_grows_with_radius(self):
        asset = procedural.bifurcation(trunk=25.0, branch=15.0)
        thin = TreeNavEnv(asset, target_node="left_out", max_steps=2,
                          device="cpu", device_radius_mm=0.05)
        thick = TreeNavEnv(asset, target_node="left_out", max_steps=2,
                           device="cpu", device_radius_mm=0.6)
        thin.reset()
        thick.reset()
        thin.sim.step(dt=2.5e-2, substeps=2)
        thick.sim.step(dt=2.5e-2, substeps=2)
        load_thin = float(np.asarray(thin.sim.wall_load_grid()).sum())
        load_thick = float(np.asarray(thick.sim.wall_load_grid()).sum())
        assert load_thick > load_thin


class TestNavEnvNativeEndpoints:
    def test_env_info_reports_native_wall_load(self):
        """wall_load_max is finite and explicitly remains in simulator units."""
        env = NavEnv(asset=procedural.straight_tube(length=TUBE_LEN, radius=TUBE_R),
                     max_steps=3, device="cpu")
        env.reset()
        _, _, _, _, info = env.step(np.array([1.0, 0.0], np.float32))
        assert "wall_load_max" in info
        assert np.isfinite([
            info["wall_load_max"], info["wall_load_sum"],
            info["wall_pressure_max"], info["wall_load_impulse"],
        ]).all()
        assert info["wall_load_max"] >= 0.0
        assert info["wall_load_units"] == "sim_units"

    def test_env_penetration_includes_device_radius(self):
        """NavEnv must subtract the configured device radius in its proxy."""
        asset = procedural.straight_tube(length=TUBE_LEN, radius=TUBE_R)
        env_default = NavEnv(asset=asset, max_steps=2, device="cpu")
        env_default.reset()
        _, max_pen_default = env_default._contact_features()

        env_thick = NavEnv(asset=asset, max_steps=2, device="cpu",
                           device_radius_mm=1.1)
        env_thick.reset()
        _, max_pen_thick = env_thick._contact_features()
        assert max_pen_default == pytest.approx(0.0, abs=1e-6)
        assert max_pen_thick == pytest.approx(0.1, abs=1e-2)

    @pytest.mark.parametrize("radius", [0.0, -0.1, float("nan"), float("inf")])
    def test_invalid_device_radius_rejected(self, radius):
        with pytest.raises(ValueError, match="radius"):
            _make_sim(1.0, radius)
    def test_diverged_step_terminates_with_finite_observation(self):
        class Sim:
            def step(self, **_):
                return None

        env = object.__new__(NavEnv)
        env.sim = Sim()
        env.substeps = 1
        env.max_insertion = 1.0
        env.max_twist = 1.0
        env.steps = 0
        env.target_s = 10.0
        env._prev_dist = 5.0
        env.success_tol = 2.5
        env.max_steps = 5
        env._tip = lambda: (float("nan"), 0.0, 0.0, 0.0)
        env._obs = lambda: np.zeros(5, dtype=np.float32)
        _, reward, terminated, truncated, info = env.step([0.0, 0.0])
        assert terminated and not truncated
        assert reward == -100.0
        assert info["diverged"] is True


class TestSuiteVersionBump:
    def test_suite_version_is_v3(self):
        from lumen.bench import SUITE_VERSION
        assert SUITE_VERSION == "lumen-bench/3"

    def test_validator_rejects_v2_scorecard(self):
        from lumen.bench import Scorecard, SUITE, validate_scorecard
        card = Scorecard(
            name="legacy-v2-submission",
            suite_version="lumen-bench/2",
            per_task=[
                {"name": t.name, "tier": t.tier, "episodes": t.episodes,
                 "success_rate": 0.0, "safe_success_rate": 0.0,
                 "mean_steps": 1.0, "max_pen": 0.0, "mean_return": 0.0}
                for t in SUITE
            ],
            overall={"success_rate": 0.0, "safe_success_rate": 0.0,
                     "max_pen": 0.0, "mean_return": 0.0},
        )
        with pytest.raises(ValueError, match="suite_version"):
            validate_scorecard(card)


class TestCalibrationSeam:
    def test_wall_load_units_documented_and_calibration_requires_data(self):
        from lumen import safety
        desc = safety.describe_wall_load_units()
        assert "sim units" in desc.lower() or "non-dimensional" in desc.lower()
        with pytest.raises(ValueError, match="paired"):
            safety.WallLoadCalibration.from_paired_measurements(
                [1.0], [0.1], provenance="invalid-single-sample"
            )

    def test_paired_calibration_converts_with_fit_diagnostics(self):
        from lumen.safety import WallLoadCalibration
        cal = WallLoadCalibration.from_paired_measurements(
            [10.0, 40.0], [0.05, 0.20], provenance="bench-test"
        )
        assert cal.convert(20.0) == pytest.approx(0.10)
        assert cal.provenance == "bench-test"
        assert cal.r_squared == pytest.approx(1.0)
        restored = WallLoadCalibration.from_dict(cal.to_dict())
        assert restored.convert(20.0) == pytest.approx(0.10)


class TestExternalComparisonContract:
    def test_metric_contract_preregistration_not_favorability(self):
        contract = json.loads(
            (ROOT / "benchmarks" / "external_comparison" / "metric_contract.json").read_text()
        )
        text = json.dumps(contract)
        assert "favorability" not in text.lower()
        assert "preregistration" in text.lower()

    def test_common_bench_emits_native_safety_per_environment(self):
        """Result schema keeps native endpoints separate; no shared threshold."""
        import importlib.util
        import sys
        from dataclasses import fields
        cb_path = ROOT / "benchmarks" / "external_comparison" / "common_bench.py"
        spec = importlib.util.spec_from_file_location("common_bench_test", cb_path)
        cb = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cb
        spec.loader.exec_module(cb)
        field_names = {f.name for f in fields(cb.EpisodeResult)}
        assert "native_safety_pass" in field_names
        assert "safety_endpoint" in field_names
        assert "safety_value" in field_names
        assert not hasattr(cb, "SAFETY_FORCE_THRESHOLD")
        result = cb.EpisodeResult(
            environment="lumen", task="tube", task_class="simple_target_navigation",
            policy="test", seed=0, success=True, steps=1,
            total_reward=0.0, final_distance=0.0,
            native_safety_pass=True,
            safety_endpoint="surface_penetration_sim_units", safety_value=0.0,
        )
        assert result.safety_endpoint == "surface_penetration_sim_units"

    def test_aggregate_excludes_unknown_cross_environment_safety(self):
        import importlib.util
        import sys
        cb_path = ROOT / "benchmarks" / "external_comparison" / "common_bench.py"
        spec = importlib.util.spec_from_file_location("common_bench_aggregate_test", cb_path)
        cb = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cb
        spec.loader.exec_module(cb)
        common = dict(
            task="branch", task_class="branch_or_arch_navigation", policy="ppo",
            seed=0, success=True, steps=10, total_reward=1.0, final_distance=0.0,
        )
        lumen = cb.EpisodeResult(
            environment="lumen", native_safety_pass=True,
            safety_endpoint="surface_penetration_sim_units", safety_value=0.1,
            unsafe_event=False, **common,
        )
        external = cb.EpisodeResult(
            environment="external", native_safety_pass=None,
            safety_endpoint="contact_force_native_units", safety_value=2.0,
            unsafe_event=None, **common,
        )
        rows = {row["environment"]: row for row in cb._aggregate([lumen, external])}
        assert rows["lumen"]["native_safety_pass_rate"] == 1.0
        assert rows["external"]["native_safety_pass_rate"] is None
        assert rows["lumen"]["safety_endpoint"] != rows["external"]["safety_endpoint"]
        assert rows["lumen"]["statistics"]["protocol"] == "lumen-stats/1"
        assert rows["lumen"]["statistics"]["metrics"]["success_rate"]["n"] == 1


def test_lumen_comparator_marks_divergence_as_crashed(tmp_path, monkeypatch):
    """The finite Gym divergence guard must reach common-bench crash accounting."""
    import argparse
    import importlib.util
    import sys

    class DivergedEnv:
        safety_max_pen = 0.3
        action_space = None

        def reset(self, *, seed=None):
            return np.zeros(5, dtype=np.float32), {}

        def step(self, action):
            return (
                np.zeros(5, dtype=np.float32),
                -100.0,
                True,
                False,
                {
                    "dist": 1e6,
                    "max_pen": 0.0,
                    "diverged": True,
                    "success": False,
                    "wall_load_max": 0.0,
                    "wall_pressure_max": 0.0,
                    "wall_load_impulse": 0.0,
                },
            )

    cb_path = ROOT / "benchmarks" / "external_comparison" / "common_bench.py"
    spec = importlib.util.spec_from_file_location("common_bench_divergence_test", cb_path)
    cb = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cb
    spec.loader.exec_module(cb)

    from lumen.envs import registration
    monkeypatch.setattr(registration, "make_nav_tube", lambda **_: DivergedEnv())
    args = argparse.Namespace(
        max_steps=1,
        policies="forward",
        tasks="nav_tube",
        episodes=1,
        seed=0,
        progress=False,
        out_dir=str(tmp_path),
        run_id="diverged",
    )
    cb.run_lumen(args)
    payload = json.loads((tmp_path / "diverged.json").read_text())
    assert payload["episodes"][0]["crashed"] is True
    assert payload["episodes"][0]["notes"]["failure_reason"] == "sim_diverged"
    assert payload["aggregate"][0]["crash_rate"] == 1.0
    assert payload["episodes"][0]["native_safety_pass"] is False

    args.max_steps = 0
    args.run_id = "zero_steps"
    cb.run_lumen(args)
    zero = json.loads((tmp_path / "zero_steps.json").read_text())
    assert zero["episodes"][0]["crashed"] is False
    assert zero["episodes"][0]["final_distance"] is None
