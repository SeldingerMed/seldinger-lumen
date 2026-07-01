"""The schematic scene viewer (`lumen play`)."""

import json

import numpy as np
import pytest

from lumen import viz


def test_render_frame_shape_and_nontrivial():
    from lumen.envs.registration import make_nav_tube
    env = make_nav_tube()
    frame = viz.render_frame(env, size=128)
    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8
    # something got drawn (wall/device/target), not a blank canvas
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 3


@pytest.mark.parametrize("scene", ["tube", "stenotic", "tree"])
def test_play_reports_frames_and_finite_safety(scene):
    s = viz.play(scene, steps=12, seed=0, size=96)
    assert s["frames"] == s["steps"] + 1          # one frame per state incl. reset
    assert np.isfinite(s["max_pen"]) and s["max_pen"] >= 0.0
    assert isinstance(s["safe"], bool)
    # safe_success can only be true when both success and safe hold
    assert s["safe_success"] == (s["success"] and s["safe"])


def test_play_writes_animation(tmp_path):
    out = tmp_path / "run"
    s = viz.play("tube", steps=8, size=96, out=str(out))
    assert (tmp_path / "run.avi").exists()
    assert (tmp_path / "run.png").exists()
    assert s["avi"].endswith("run.avi")


def test_play_matches_bench_safety_on_tree():
    # the hard tier reaches the target but breaches the wall — the viewer must report
    # the same unsafe outcome the benchmark scores (guards the shared safety seam).
    s = viz.play("tree", steps=60, seed=200, size=96)
    assert s["success"] is True
    assert s["safe"] is False
    assert s["max_pen"] > 0.3


def test_cli_play_smoke(tmp_path, capsys):
    from lumen.cli import play_main
    play_main(["tube", "--steps", "6", "--size", "80", "--out", str(tmp_path / "p")])
    payload = json.loads(capsys.readouterr().out)
    assert payload["scene"] == "tube"
    assert payload["frames"] == payload["steps"] + 1


def test_train_saves_policy_and_play_loads_it(tmp_path, capsys):
    # train (tiny CEM) -> .npz, then play that saved policy end-to-end: visualize a
    # trained agent. Guards the train->play seam.
    from lumen.cli import train_main
    out = tmp_path / "policy.npz"
    train_main(["tube", "--pop", "8", "--iters", "3", "--out", str(out)])
    payload = json.loads(capsys.readouterr().out)
    assert out.exists()
    assert payload["policy"].endswith("policy.npz")

    s = viz.play("tube", policy=str(out), steps=20, size=80)
    assert s["frames"] == s["steps"] + 1
    assert isinstance(s["success"], bool)


def test_play_rejects_unknown_policy():
    with pytest.raises(ValueError):
        viz.play("tube", policy="nonsense", steps=2, size=64)
