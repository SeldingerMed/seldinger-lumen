from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.external_comparison import (
    eval_comparator_contact,
    xseries_chart,
    xseries_compare,
)
from tools.xseries import clip_matched, xs_common

CMP = "cath" + "sim"


def test_contact_probe_uses_copy_without_mutating_live_physics():
    live_contact_force_calls = []
    copy_contact_force_calls = []

    class FakeContact:
        def __init__(self, geom1, geom2, dist):
            self.geom1 = geom1
            self.geom2 = geom2
            self.dist = dist

    class FakeModel:
        nbody = 3
        geom_bodyid = [0, 1, 2]

        def id2name(self, idx, kind):
            return ["phantom/wall", "guidewire/shaft_0", "guidewire/tip/0"][idx]

    class FakeData:
        def __init__(self, is_copy: bool):
            self.is_copy = is_copy
            self.ncon = 2
            self.contact = [FakeContact(0, 1, 0.0005), FakeContact(0, 2, 0.0005)]

        def contact_force(self, i):
            if self.is_copy:
                copy_contact_force_calls.append(i)
            else:
                live_contact_force_calls.append(i)
            val = 0.75 if i == 0 else 1.25
            return np.array([[val, 0.0, 0.0], [0.0, 0.0, 0.0]])

    class FakePhysics:
        def __init__(self, is_copy=False):
            self.model = FakeModel()
            self.data = FakeData(is_copy=is_copy)

        def copy(self, share_model=True):
            return FakePhysics(is_copy=True)

    class FakeEnv:
        def __init__(self):
            self._env = SimpleNamespace(physics=FakePhysics(False), _task=SimpleNamespace(get_contact_forces=lambda *a: None))
            self.unwrapped = self

    probe = eval_comparator_contact.contact_probe(FakeEnv())
    tip, wire = probe()
    assert live_contact_force_calls == []
    assert copy_contact_force_calls == [0, 1]
    assert tip == pytest.approx(1.25)
    assert wire == pytest.approx(1.25)


def test_eval_comparator_contact_validations_and_per_episode_spl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError, match="episodes"):
        eval_comparator_contact.main(["--models", str(tmp_path), "--episodes", "0", "--out", str(tmp_path / "o.json")])
    with pytest.raises(ValueError, match="max-steps"):
        eval_comparator_contact.main(["--models", str(tmp_path), "--max-steps", "0", "--out", str(tmp_path / "o.json")])
    with pytest.raises(ValueError, match="unique"):
        eval_comparator_contact.main(["--models", str(tmp_path), "--seeds", "0,0", "--out", str(tmp_path / "o.json")])

    # Per-episode SPL differs from mean-path SPL when path lengths vary
    rows = [
        {
            "seed": 10000, "success": True, "steps": 10, "time_to_target_s": None,
            "peak_tip_contact_force_N": 0.2, "mean_tip_contact_force_N": 0.1,
            "peak_wire_contact_force_N": 0.2, "eval_wall_s": 0.0,
            "safe_success_0.5N": True, "safe_success_1.0N": True, "safe_success_1.5N": True,
            f"{CMP}_native": {"force_mean": 0.5, "force_max": 1.0, "episode_length": 10,
                                "safety": 1.0, "total_distance_cm": 15.73, "success": 1},
        },
        {
            "seed": 10001, "success": False, "steps": 50, "time_to_target_s": None,
            "peak_tip_contact_force_N": 0.4, "mean_tip_contact_force_N": 0.2,
            "peak_wire_contact_force_N": 0.4, "eval_wall_s": 0.0,
            "safe_success_0.5N": False, "safe_success_1.0N": False, "safe_success_1.5N": False,
            f"{CMP}_native": {"force_mean": 0.5, "force_max": 1.0, "episode_length": 50,
                                "safety": 1.0, "total_distance_cm": 100.0, "success": 0},
        },
    ]
    s = eval_comparator_contact.summarize(rows)
    # Episode 0 contributes 1.0, episode 1 contributes 0.0 -> true SPL = 0.5 (not 0.5 * 15.73 / 57.865)
    assert s[f"{CMP}_native"]["spl"] == pytest.approx(0.5)
    assert s["mean_steps_success"] == pytest.approx(10.0)
    assert s["eval_steps_per_s"] is None


def test_eval_comparator_contact_continues_after_seed_failure_and_closes_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    closed = []

    class BoomEnv:
        def __init__(self):
            self.unwrapped = self

        def reset(self, seed=None):
            raise RuntimeError("reset boom")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(eval_comparator_contact, "make_env", lambda task, max_steps, seed: BoomEnv())
    with pytest.raises(RuntimeError, match="reset boom"):
        eval_comparator_contact.evaluate(object(), "phantom3_bca", 1, 10)
    assert closed == [True]

    # Seed 0 fails load, seed 1 missing -> both recorded
    (tmp_path / f"{CMP}-phantom3_bca-ppo-main-s0.zip").write_text("corrupt")
    out_file = tmp_path / "sub" / "contact.json"
    import sys

    class FakePPO:
        @staticmethod
        def load(path, device="cpu"):
            raise ValueError("bad zip")

    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=FakePPO))
    eval_comparator_contact.main([
        "--models", str(tmp_path), "--seeds", "0,1", "--episodes", "2", "--max-steps", "20", "--out", str(out_file)
    ])
    payload = json.loads(out_file.read_text())
    assert payload["per_seed"]["0"]["status"] == "failed"
    assert "bad zip" in payload["per_seed"]["0"]["error"]
    assert payload["per_seed"]["1"]["status"] == "missing_model"
    assert "not N" in payload[f"{CMP}_native_metrics"]
    assert "< 20" in payload[f"{CMP}_native_metrics"]


def test_xseries_compare_handles_failed_runs_none_safety_projected_and_validation(tmp_path: Path):
    bench = tmp_path / "bench"
    bench.mkdir()
    # Lumen seed 0 ok, seed 1 failed with safety_value=None and train_elapsed_sec=0
    (bench / "lumen-nav_tree_branch-ppo-main-s0.json").write_text(json.dumps({
        "task": "nav_tree_branch",
        "aggregate": [{"success_rate": 1.0, "native_safety_pass_rate": 1.0, "max_wall_load": 5.0, "steps_per_second": 200.0}],
        "episodes": [{"steps": 40, "success": True, "safety_value": 0.1}],
        "extra": {"seed": 0, "timesteps": 600000, "train_elapsed_sec": 3600.0, "eval_episodes": 1, "eval_seed_start": 10000,
                   "runs": [{"seed": 0, "failed": False, "resumed_from_step": 0}], "failed": False},
    }))
    (bench / "lumen-nav_tree_branch-ppo-main-s1.json").write_text(json.dumps({
        "task": "nav_tree_branch",
        "aggregate": [{"success_rate": 0.0, "native_safety_pass_rate": None, "max_wall_load": None, "steps_per_second": 0.0}],
        "episodes": [{"steps": 0, "success": False, "safety_value": None, "extra": {"failed": True}}],
        "extra": {"seed": 1, "timesteps": 600000, "train_elapsed_sec": 0.0, "eval_episodes": 1, "eval_seed_start": 10000,
                   "runs": [{"seed": 1, "failed": True}], "failed": True, "error": "boom"},
    }))
    # Comparator seed 0 resumed ok, seed 5 failed
    (bench / f"{CMP}-phantom3_bca-ppo-main-s0.json").write_text(json.dumps({
        "task": "phantom3_bca",
        "aggregate": [{"success_rate": 0.8, "steps_per_second": 15.0}],
        "episodes": [{"steps": 100, "success": True, "safety_value": 0.5}],
        "extra": {"seed": 0, "timesteps": 600000, "train_elapsed_sec": 3600.0, "eval_episodes": 1, "eval_seed_start": 10000,
                   "runs": [{"seed": 0, "failed": False, "resumed_from_step": 300000}], "failed": False},
    }))
    (bench / f"{CMP}-phantom3_bca-ppo-main-s5.json").write_text(json.dumps({
        "task": "phantom3_bca",
        "aggregate": [{"success_rate": 0.0, "steps_per_second": 0.0}],
        "episodes": [{"steps": 0, "success": False, "safety_value": None}],
        "extra": {"seed": 5, "timesteps": 600000, "train_elapsed_sec": 0.0, "eval_episodes": 1, "eval_seed_start": 10000,
                   "runs": [{"seed": 5, "failed": True}], "failed": True, "error": "oom"},
    }))

    # Missing --contact file raises
    with pytest.raises(FileNotFoundError):
        xseries_compare.main(["--bench", str(bench), "--contact", str(tmp_path / "nope.json"), "--out", str(tmp_path / "c")])

    # Bad task in contact raises
    bad_contact = tmp_path / "bad_contact.json"
    bad_contact.write_text(json.dumps({"task": "phantom3_lcca", "episodes": 1, "per_seed": {}}))
    with pytest.raises(ValueError, match="task mismatch"):
        xseries_compare.main(["--bench", str(bench), "--contact", str(bad_contact), "--out", str(tmp_path / "c")])

    # Bad episodes in contact raises
    bad_ep_contact = tmp_path / "bad_ep.json"
    bad_ep_contact.write_text(json.dumps({"task": "phantom3_bca", "episodes": 10, "per_seed": {}}))
    with pytest.raises(ValueError, match="episode count mismatch"):
        xseries_compare.main(["--bench", str(bench), "--contact", str(bad_ep_contact), "--out", str(tmp_path / "c")])

    # Default --seeds is 0,1,5; works without --contact and keeps known raw success
    out_prefix = tmp_path / "cmp"
    xseries_compare.main(["--bench", str(bench), "--out", str(out_prefix)])
    res = json.loads((tmp_path / "cmp.json").read_text())
    md = (tmp_path / "cmp.md").read_text()
    assert res["training_seeds"] == [0, 1, 5]
    assert res["lumen"]["per_seed"][1]["status"] == "failed"
    assert res[CMP]["per_seed"][2]["status"] == "failed"
    assert res[CMP]["summary"]["success_rate"]["mean"] == pytest.approx(0.8)
    assert res[CMP]["any_resumed_train_hours_projected"] is True
    assert "(projected)" in md
    assert "failed" in md

    # Chart renders cleanly on partial/failed comparison JSON
    chart_png = tmp_path / "chart.png"
    xseries_chart.main(["--compare", str(tmp_path / "cmp.json"), "--out", str(chart_png)])
    assert chart_png.exists() and chart_png.stat().st_size > 0


def test_cached_rollout_keys_on_extra_fn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = []

    class Env:
        substeps = 4

    def fake_rollout(env, policy, max_steps, seed=0, extra_fn=None):
        calls.append(extra_fn)
        return [SimpleNamespace(nodes=[0, 1], tag=getattr(extra_fn, "__name__", None))]

    def pen(env):
        return {"pen": 1}

    monkeypatch.setattr(xs_common, "out_root", lambda: tmp_path)
    monkeypatch.setattr(xs_common, "rollout", fake_rollout)
    (f0, _), p0 = xs_common.cached_rollout("arch", Env, None, 5)
    (f1, _), p1 = xs_common.cached_rollout("arch", Env, None, 5, extra_fn=pen)
    (f2, _), p2 = xs_common.cached_rollout("arch", Env, None, 5, extra_fn=pen)
    assert p0 != p1 and p1 == p2
    assert f0[0].tag is None and f1[0].tag == "pen" and f2[0].tag == "pen"
    assert len(calls) == 2


def test_clip_matched_validates_inputs(tmp_path: Path):
    with pytest.raises(ValueError, match="--slow"):
        clip_matched.main(["--episodes", str(tmp_path), "--slow", "0", "--out", str(tmp_path / "m")])
    with pytest.raises(FileNotFoundError, match="--compare"):
        clip_matched.main(["--episodes", str(tmp_path), "--compare", str(tmp_path / "missing.json"),
                           "--out", str(tmp_path / "m")])
    cmp = tmp_path / "c.json"
    cmp.write_text(json.dumps({"training_seeds": [0, 1, 5]}))
    with pytest.raises(FileNotFoundError, match="matched recordings missing"):
        clip_matched.main(["--episodes", str(tmp_path), "--compare", str(cmp), "--out", str(tmp_path / "m")])
    assert not (tmp_path / "m").exists()
