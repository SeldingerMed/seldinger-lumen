from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.external_comparison import (
    eval_comparator_contact,
    xseries_compare,
)

# Rendering dependencies (matplotlib, Pillow) are optional extras, not part of the CI install.

CMP = "cath" + "sim"


def test_contact_probe_uses_copy_without_mutating_live_physics(monkeypatch):
    calls = []          # (fn, data id): every mujoco call must hit the copy, never the live data

    class FakeContact:
        def __init__(self, geom1, geom2, dist):
            self.geom1, self.geom2, self.dist = geom1, geom2, dist

    class FakeModel:
        nbody = 4
        geom_bodyid = [0, 1, 2, 3]
        ptr = "model"

        def id2name(self, idx, kind):
            return ["phantom/wall", "guidewire/shaft_0", "guidewire/tip/0", "sheath/body"][idx]

    class FakeData:
        def __init__(self, tag):
            self.ptr = tag
            self.ncon = 3
            self.contact = [FakeContact(0, 1, 0.0005), FakeContact(0, 2, 0.0005), FakeContact(3, 0, 0.0005)]

    class FakePhysics:
        def __init__(self, tag="live"):
            self.model, self.data = FakeModel(), FakeData(tag)

        def copy(self, share_model=True):
            return FakePhysics("copy")

    force = {0: 0.75, 1: 1.25, 2: 9.0}       # contact 2 is phantom-sheath: not wire force

    class FakeMujoco:
        def __getattr__(self, name):
            def fn(m, d, *rest):
                calls.append((name, d))
                if name == "mj_contactForce":
                    i, wrench = rest
                    wrench[:] = 0.0
                    wrench[0] = force[i]
            return fn

    monkeypatch.setattr(eval_comparator_contact, "_mujoco", lambda: FakeMujoco())

    class FakeEnv:
        def __init__(self):
            self._env = SimpleNamespace(physics=FakePhysics(), _task=SimpleNamespace(get_contact_forces=lambda *a: None))
            self.unwrapped = self

    tip, wire = eval_comparator_contact.contact_probe(FakeEnv())()
    assert {d for _, d in calls} == {"copy"}
    assert [r for r in calls if r[0] == "mj_contactForce"] == [("mj_contactForce", "copy")] * 2
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
    rows, _ = eval_comparator_contact.evaluate(object(), "phantom3_bca", 2, 10)   # each episode recorded
    assert closed == [True, True]

    def bad_close():
        closed.append(True)
        raise OSError("close boom")
    monkeypatch.setattr(BoomEnv, "close", lambda self: bad_close())
    rows2, _ = eval_comparator_contact.evaluate(object(), "phantom3_bca", 2, 10)   # teardown error tolerated
    assert len(rows2) == 2 and "reset boom" in rows2[1]["error"]
    monkeypatch.setattr(BoomEnv, "close", lambda self: closed.append(True))
    assert [r["status"] for r in rows] == ["failed", "failed"] and "reset boom" in rows[0]["error"]
    summ = eval_comparator_contact.summarize(rows)
    assert summ["failed_episodes"] == 2 and summ["success_rate"] == 0.0
    assert summ["peak_tip_contact_force_N_max"] is None

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

    # A seed whose every episode raised is a failed seed, not a 0% result
    monkeypatch.setattr(FakePPO, "load", staticmethod(lambda path, device="cpu": object()))
    eval_comparator_contact.main([
        "--models", str(tmp_path), "--seeds", "0", "--episodes", "2", "--max-steps", "20", "--out", str(out_file)
    ])
    p0 = json.loads(out_file.read_text())["per_seed"]["0"]
    assert p0["status"] == "failed" and "all 2 episodes failed" in p0["error"]


def test_eval_comparator_contact_time_to_target_is_first_reach(monkeypatch: pytest.MonkeyPatch):
    class ReachEnv:          # tip within 4 mm from step 3 on; the episode only truncates at step 10
        def __init__(self):
            self.unwrapped, self.t = self, 0

        def reset(self, seed=None):
            self.t = 0
            return np.zeros(3), {}

        def step(self, action):
            self.t += 1
            d = 0.01 if self.t < 3 else 0.002
            return np.zeros(3), 0.0, False, self.t >= 10, {"head_pos": [d, 0, 0], "target_pos": [0, 0, 0]}

        def close(self):
            pass

    class Policy:
        def predict(self, obs, deterministic=True):
            return np.zeros(2), None

    monkeypatch.setattr(eval_comparator_contact, "make_env", lambda task, max_steps, seed: ReachEnv())
    monkeypatch.setattr(eval_comparator_contact, "seed_comparator_rng", lambda env, seed: None)
    monkeypatch.setattr(eval_comparator_contact, "control_dt", lambda env: 0.1)
    monkeypatch.setattr(eval_comparator_contact, "contact_probe", lambda env: (lambda: (0.0, 0.0)))
    rows, dt = eval_comparator_contact.evaluate(Policy(), "phantom3_bca", 1, 20)
    assert rows[0]["steps"] == 10 and rows[0]["reach_step"] == 3
    assert rows[0]["time_to_target_s"] == pytest.approx(0.3)
    assert eval_comparator_contact.summarize(rows)["mean_steps_success"] == pytest.approx(3)


def test_font_falls_back_when_the_mac_font_is_missing(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("PIL")
    from tools.xseries import xs_common
    monkeypatch.setattr(xs_common, "FONT", "/nonexistent/font.ttc")
    assert xs_common.font(18) is not None


def _bench_result(task, seed, *, ok, sr=1.0, resumed=0, elapsed=3600.0, model_dir="/m", episodes=None,
                  aggregate=None, error="boom"):
    """A result JSON in the schema train_sb3.py writes (one training seed per file)."""
    model = f"{model_dir}/{task}-s{seed}.zip"
    run = {"seed": seed, "model_path": model, "status": "completed" if ok else "failed",
           "train_elapsed_sec": elapsed if ok else 0.0, "elapsed_sec": elapsed, "eval_episodes": 1,
           "resumed_from_step": resumed}
    if not ok:
        run.update({"failure_stage": "train", "error": f"RuntimeError('{error}')"})
    return {"run_id": f"{task}-s{seed}", "environment": "lumen" if task.startswith("nav") else CMP,
            "task_specs": [{"name": task, "task_class": "branch_or_arch_navigation"}],
            "aggregate": aggregate or [{"success_rate": sr, "native_safety_pass_rate": 1.0, "max_wall_load": 5.0,
                                        "steps_per_second": 200.0}],
            "episodes": episodes or [{"steps": 40, "success": True, "safety_value": 0.1}],
            "extra": {"algo": "ppo", "timesteps": 600000, "seed": seed, "seeds": [seed],
                      "failed_seeds": [] if ok else [seed], "eval_episodes_per_seed": 1,
                      "evaluation_seed_schedule": [10000], "runs": [run],
                      "train_elapsed_sec": run["train_elapsed_sec"],
                      "failure_stages": {} if ok else {str(seed): "train"}}}


def test_xseries_compare_handles_failed_runs_none_safety_projected_and_validation(tmp_path: Path):
    bench = tmp_path / "bench"
    bench.mkdir()
    w = lambda name, j: (bench / name).write_text(json.dumps(j))  # noqa: E731
    w("lumen-nav_tree_branch-ppo-main-s0.json", _bench_result("nav_tree_branch", 0, ok=True))
    w("lumen-nav_tree_branch-ppo-main-s1.json", _bench_result(
        "nav_tree_branch", 1, ok=False, episodes=[{"steps": 0, "success": False, "safety_value": None}],
        aggregate=[{"success_rate": 0.0, "native_safety_pass_rate": None, "max_wall_load": None,
                    "steps_per_second": 0.0}]))
    w(f"{CMP}-phantom3_bca-ppo-main-s0.json", _bench_result("phantom3_bca", 0, ok=True, sr=0.8, resumed=300000))
    w(f"{CMP}-phantom3_bca-ppo-main-s5.json", _bench_result("phantom3_bca", 5, ok=False, error="oom"))

    # Missing --contact file raises
    with pytest.raises(FileNotFoundError):
        xseries_compare.main(["--bench", str(bench), "--contact", str(tmp_path / "nope.json"), "--out", str(tmp_path / "c")])
    # Wrong task in contact raises
    bad_contact = tmp_path / "bad_contact.json"
    bad_contact.write_text(json.dumps({"task": "phantom3_lcca", "episodes": 1, "per_seed": {}}))
    with pytest.raises(ValueError, match="task mismatch"):
        xseries_compare.main(["--bench", str(bench), "--contact", str(bad_contact), "--out", str(tmp_path / "c")])
    # Contact scored with a different episode count than the training evaluation raises
    bad_ep_contact = tmp_path / "bad_ep.json"
    bad_ep_contact.write_text(json.dumps({"task": "phantom3_bca", "episodes": 10, "eval_seed_start": 10000,
                                          "per_seed": {}}))
    with pytest.raises(ValueError, match="episode count mismatch"):
        xseries_compare.main(["--bench", str(bench), "--contact", str(bad_ep_contact), "--out", str(tmp_path / "c")])
    # Contact that scored a different model file for the seed raises
    wrong_model = tmp_path / "wrong_model.json"
    wrong_model.write_text(json.dumps({"task": "phantom3_bca", "episodes": 1, "eval_seed_start": 10000,
                                       "per_seed": {"0": {"status": "ok", "model": "/other/model.zip",
                                                          "summary": {}}}}))
    with pytest.raises(ValueError, match="not this run's model"):
        xseries_compare.main(["--bench", str(bench), "--contact", str(wrong_model), "--out", str(tmp_path / "c")])

    # Contact that scored a model whose bytes no longer match the run's model file raises
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "phantom3_bca-s0.zip").write_bytes(b"retrained")
    j0 = json.loads((bench / f"{CMP}-phantom3_bca-ppo-main-s0.json").read_text())
    j0["extra"]["runs"][0]["model_path"] = str(model_dir / "phantom3_bca-s0.zip")
    w(f"{CMP}-phantom3_bca-ppo-main-s0.json", j0)
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"task": "phantom3_bca", "episodes": 1, "eval_seed_start": 10000,
                                 "per_seed": {"0": {"status": "ok", "model": str(model_dir / "phantom3_bca-s0.zip"),
                                                    "model_sha256": "0" * 64, "summary": {}}}}))
    with pytest.raises(ValueError, match="sha256 differs"):
        xseries_compare.main(["--bench", str(bench), "--contact", str(stale), "--out", str(tmp_path / "c")])
    (model_dir / "phantom3_bca-s0.zip").unlink()
    with pytest.raises(ValueError, match="not on disk"):
        xseries_compare.main(["--bench", str(bench), "--contact", str(stale), "--out", str(tmp_path / "c")])

    # A matching sha256 is reported as verified; a pre-hashing contact file is kept but marked
    import hashlib
    (model_dir / "phantom3_bca-s0.zip").write_bytes(b"frozen")
    summ = {"success_rate": 0.8, "safe_success_rate_1.0N": 0.0, "safe_success_rate_0.5N": 0.0,
            "safe_success_rate_1.5N": 0.0, "peak_tip_contact_force_N_max": 3.0, "peak_tip_contact_force_N_mean": 2.0,
            "mean_tip_contact_force_N": 0.5, "mean_time_to_target_s": 0.4, "distinct_eval_outcomes": 1,
            "episodes": 1, f"{CMP}_native": {"force_mean": 0.1}}
    for sha, check in ((hashlib.sha256(b"frozen").hexdigest(), "sha256"), (None, "filename only")):
        rec = {"status": "ok", "model": str(model_dir / "phantom3_bca-s0.zip"), "summary": summ}
        if sha:
            rec["model_sha256"] = sha
        cf = tmp_path / "contact_ok.json"
        cf.write_text(json.dumps({"task": "phantom3_bca", "episodes": 1, "eval_seed_start": 10000,
                                  "per_seed": {"0": rec}}))
        xseries_compare.main(["--bench", str(bench), "--contact", str(cf), "--out", str(tmp_path / "v")])
        rv = json.loads((tmp_path / "v.json").read_text())
        assert rv[CMP]["per_seed"][0]["contact_model_check"].startswith(check)
        assert rv[CMP]["contact_model_unverified_seeds"] == ([] if sha else [0])
        assert ("filename only" in (tmp_path / "v.md").read_text()) is (sha is None)
    # A seed the contact pass failed on is reported, not silently averaged over fewer seeds
    cf.write_text(json.dumps({"task": "phantom3_bca", "episodes": 1, "eval_seed_start": 10000,
                              "per_seed": {"0": {"status": "failed", "error": "mj boom"}}}))
    xseries_compare.main(["--bench", str(bench), "--contact", str(cf), "--out", str(tmp_path / "v")])
    rv = json.loads((tmp_path / "v.json").read_text())
    assert rv[CMP]["contact_missing_seeds"] == [0]
    assert rv[CMP]["per_seed"][0]["contact_error"] == "mj boom"
    vmd = (tmp_path / "v.md").read_text()
    assert "(contact failed)" in vmd and "no result for seed(s) 0" in vmd
    # A pass in which every episode raised keeps the training-eval success and is reported
    allfail = dict(summ, success_rate=0.0, failed_episodes=1, episodes=1)
    cf.write_text(json.dumps({"task": "phantom3_bca", "episodes": 1, "eval_seed_start": 10000,
                              "per_seed": {"0": {"status": "ok", "model": str(model_dir / "phantom3_bca-s0.zip"),
                                                 "model_sha256": hashlib.sha256(b"frozen").hexdigest(),
                                                 "summary": allfail}}}))
    xseries_compare.main(["--bench", str(bench), "--contact", str(cf), "--out", str(tmp_path / "v")])
    rv = json.loads((tmp_path / "v.json").read_text())
    assert rv[CMP]["per_seed"][0]["success_rate"] == pytest.approx(0.8)
    assert rv[CMP]["contact_missing_seeds"] == [0]
    assert "all_episodes_failed" in (tmp_path / "v.md").read_text()

    # Default --seeds is 0,1,5; works without --contact and keeps known raw success
    out_prefix = tmp_path / "cmp"
    xseries_compare.main(["--bench", str(bench), "--out", str(out_prefix)])
    res = json.loads((tmp_path / "cmp.json").read_text())
    md = (tmp_path / "cmp.md").read_text()
    assert res["training_seeds"] == [0, 1, 5]
    assert res["lumen"]["per_seed"][1]["status"] == "failed"
    assert "boom" in res["lumen"]["per_seed"][1]["error"]
    assert res[CMP]["per_seed"][1]["status"] == "not_finished"          # seed 1 never ran
    assert res[CMP]["per_seed"][2]["status"] == "failed"
    assert res[CMP]["summary"]["success_rate"]["mean"] == pytest.approx(0.8)
    assert res[CMP]["summary"]["success_rate"]["n"] == 1
    assert res[CMP]["any_resumed_train_hours_projected"] is True
    assert "(projected)" in md
    assert "failed" in md
    raw_row = next(line for line in md.splitlines() if line.startswith("| Raw success"))
    assert raw_row.count("80%") == 1      # an n=1 mean is printed once, without a duplicate

    # A multi-seed result file is rejected with a clear message, not a TypeError
    multi = _bench_result("nav_tree_branch", 2, ok=True)
    multi["extra"]["seed"] = None
    multi["extra"]["runs"].append(dict(multi["extra"]["runs"][0], seed=3))
    w("lumen-nav_tree_branch-ppo-main-s2.json", multi)
    with pytest.raises(ValueError, match="multi-seed result file"):
        xseries_compare.main(["--bench", str(bench), "--out", str(tmp_path / "m")])
    (bench / "lumen-nav_tree_branch-ppo-main-s2.json").unlink()

    # Chart renders cleanly on partial/failed comparison JSON
    pytest.importorskip("matplotlib")
    from benchmarks.external_comparison import xseries_chart
    chart_png = tmp_path / "chart.png"
    xseries_chart.main(["--compare", str(tmp_path / "cmp.json"), "--out", str(chart_png)])
    assert chart_png.exists() and chart_png.stat().st_size > 0


def test_end_cards_handle_missing_means():
    pytest.importorskip("PIL")
    from tools.xseries import clip_matched, clip_recap
    none = {"mean": None, "lo": None, "hi": None, "n": 0}
    cmp = {"lumen": {"summary": {"success_rate": none, "safe_success_rate": none, "train_hours": none}},
           CMP: {"summary": {"success_rate": {"mean": 1.0, "n": 1}, "safe_success_rate": none,
                             "train_hours": {"mean": 16.7, "n": 1}}}}
    line = clip_matched.summary_line(cmp)
    assert "Lumen n/a" in line and "100% (n=1)" in line
    lines = clip_recap.head_to_head_lines(cmp)
    assert any("16.7 h (n=1)" in t for t, _, _ in lines)
    empty = {"lumen": {"summary": {"success_rate": none, "safe_success_rate": none, "train_hours": none}},
             CMP: {"summary": {"success_rate": none, "safe_success_rate": none, "train_hours": none}}}
    assert clip_matched.summary_line(empty) is None and clip_recap.head_to_head_lines(empty) is None


def test_checkpoint_identity_rejects_other_config(tmp_path: Path):
    import argparse

    from benchmarks.external_comparison import train_sb3
    args = argparse.Namespace(environment=CMP, task="phantom3_bca", algo="ppo", timesteps=600000,
                              ppo_n_steps=2048, batch_size=64, max_steps=300)
    train_sb3._check_checkpoint_identity(tmp_path, train_sb3._checkpoint_identity(args, 0))   # adopts
    train_sb3._check_checkpoint_identity(tmp_path, train_sb3._checkpoint_identity(args, 0))   # same run: ok
    with pytest.raises(ValueError, match="different run configuration"):
        train_sb3._check_checkpoint_identity(tmp_path, train_sb3._checkpoint_identity(args, 1))
    args.ppo_n_steps = 1024
    with pytest.raises(ValueError, match="different run configuration"):
        train_sb3._check_checkpoint_identity(tmp_path, train_sb3._checkpoint_identity(args, 0))


def test_cached_rollout_keys_on_extra_fn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("PIL")
    from tools.xseries import clip_matched, xs_common  # noqa: F401
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

    def pen(env):  # noqa: F811  (a different function under the same name)
        return {"pen": 2, "w": 0}

    (_, _), p3 = xs_common.cached_rollout("arch", Env, None, 5, extra_fn=pen)
    assert p3 != p1 and len(calls) == 3
    (_, _), p5 = xs_common.cached_rollout("arch", Env, None, 7, extra_fn=pen)
    assert p5 != p3 and len(calls) == 4          # different max_steps: different cache entry
    calls.pop()

    monkeypatch.setattr(xs_common, "_rollout_code_version", lambda: "changed0")
    (_, _), p4 = xs_common.cached_rollout("arch", Env, None, 5, extra_fn=pen)
    assert p4 != p3 and len(calls) == 4          # recording code changed: cache not reused


def test_clip_matched_validates_inputs(tmp_path: Path):
    pytest.importorskip("PIL")
    from tools.xseries import clip_matched, xs_common  # noqa: F401
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
