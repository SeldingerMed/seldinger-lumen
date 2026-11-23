import json
from pathlib import Path

import pytest

from benchmarks.external_comparison import xseries_v2_compare as v2


def _train(path: Path, seed: int, elapsed: float, resumed: int = 0, n_envs: int = 1):
    path.write_text(json.dumps({"extra": {"timesteps": 600000, "train_elapsed_sec": elapsed, "n_envs": n_envs,
                                          "runs": [{"seed": seed, "resumed_from_step": resumed}]}}))


def _curve(path: Path, seed: int, task: str, reach, pts):
    path.write_text(json.dumps({"identity": {"training_seed": seed, "task": f"phantom3_{task}"},
                                "first_reach_steps": reach,
                                "points": [{"steps": s, "success_rate": r, "exact": e, "eval_env_steps": 100,
                                            "eval_wall_s": 10.0} for s, r, e in pts]}))


def test_time_to_target_not_reached_and_identity(tmp_path: Path):
    spec = []
    for seed, elapsed, reach in ((0, 6000.0, 100000), (1, 6000.0, 200000), (5, 6000.0, None)):
        t, c = tmp_path / f"t{seed}.json", tmp_path / f"c{seed}.json"
        _train(t, seed, elapsed)
        _curve(c, seed, "bca", reach, [(50000, 0.2, False), (100000, 0.95 if seed == 0 else 0.5, seed == 0),
                                       (200000, 0.92 if seed == 1 else 0.8, seed == 1)])
        spec.append({"sim": "lumen", "config": "single", "task": "bca", "seed": seed, "train": str(t), "curve": str(c)})
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    v2.main(["--spec", str(tmp_path / "spec.json"), "--out", str(tmp_path / "out")])
    res = json.loads((tmp_path / "out.json").read_text())
    g = res["summary"]["lumen/single/bca"]
    assert g["reached"] == 2 and g["runs"] == 3
    # 600000 steps / 6000 s = 100 env-steps/s -> 100k steps = 1000 s, 200k = 2000 s; mean 1500 s
    assert g["hours_to_target"]["mean"] == pytest.approx(1500 / 3600)
    r5 = next(r for r in res["runs"] if r["seed"] == 5)
    assert r5["reached"] is False and r5["hours_to_target"] is None and r5["best_success"] == pytest.approx(0.8)
    md = (tmp_path / "out.md").read_text()
    assert "2/3" in md and "not reached" in md
    bad = [dict(spec[0], seed=1)]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="is for seed 0"):
        v2.main(["--spec", str(tmp_path / "bad.json"), "--out", str(tmp_path / "bad")])


def test_chart_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    from benchmarks.external_comparison import xseries_v2_chart
    test_time_to_target_not_reached_and_identity(tmp_path)
    png = tmp_path / "chart.png"
    xseries_v2_chart.main(["--compare", str(tmp_path / "out.json"), "--out", str(png)])
    assert png.stat().st_size > 10_000
