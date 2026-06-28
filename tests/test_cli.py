"""Installed command entry points for first-run workflows."""

import json
from importlib.metadata import metadata
from importlib.metadata import entry_points

import numpy as np
import pytest

from lumen.assets import procedural
from lumen.data import Episode, EpisodeMeta, Outcome, Step, iter_index_records, load_step_record
from lumen.sensors.carm import CArm


def test_distribution_metadata_matches_public_project_name():
    assert metadata("seldinger-lumen")["Name"] == "seldinger-lumen"


def test_pyproject_exposes_first_run_console_scripts():
    scripts = {
        ep.name: ep.value
        for ep in entry_points(group="console_scripts")
        if ep.name == "lumen" or ep.name.startswith("lumen-")
    }

    assert scripts == {
        "lumen": "lumen.cli:main",
        "lumen-hardware": "lumen.cli:hardware_main",
        "lumen-benchmark": "lumen.cli:benchmark_main",
        "lumen-render-fluoro": "lumen.cli:render_fluoro_main",
        "lumen-capture": "lumen.cli:capture_main",
        "lumen-replay": "lumen.cli:replay_main",
        "lumen-index": "lumen.cli:index_main",
        "lumen-calibrate": "lumen.cli:calibrate_main",
    }


def test_umbrella_cli_dispatches_workflows(capsys):
    from lumen.cli import main

    main(["hardware"])

    payload = json.loads(capsys.readouterr().out)
    assert "newton_available" in payload
    assert "backend_validated" in payload


def test_umbrella_cli_subcommand_help_uses_subcommand_prog(capsys):
    from lumen.cli import main

    with pytest.raises(SystemExit) as seen:
        main(["index", "--help"])

    assert seen.value.code == 0
    out = capsys.readouterr().out
    assert "usage: lumen index" in out
    assert "--check-sidecars" in out


def test_replay_cli_handles_missing_root_without_warning(tmp_path, capsys):
    import warnings

    from lumen.cli import replay_main

    missing = tmp_path / "missing"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        replay_main([str(missing)])

    out = capsys.readouterr().out
    assert "run examples/capture_episode.py first" in out
    assert seen == []


def test_render_fluoro_cli_writes_preview_artifacts(tmp_path, capsys):
    from lumen.cli import render_fluoro_main

    out = tmp_path / "fluoro.png"
    render_fluoro_main([str(out)])

    msg = capsys.readouterr().out
    assert "tip keypoint view0=" in msg
    assert out.exists()
    assert (tmp_path / "fluoro_lateral.png").exists()
    assert (tmp_path / "fluoro_device_mask.png").exists()
    assert (tmp_path / "fluoro_vessel_mask.png").exists()
    assert (tmp_path / "fluoro_biplanar.avi").exists()


def test_index_cli_writes_cv_jsonl_for_case_bundle(tmp_path, capsys):
    from lumen.cli import index_main

    carm = CArm.looking_at([0.0, 0.0, 40.0], axis=(1.0, 0.0, 0.0), nu=16, nv=16)
    ep = Episode(
        meta=EpisodeMeta(
            asset_ref="asset.json",
            device={"guidewire": {"radius": 0.2}},
            sensor={"modality": "fluoro", "nu": 16, "nv": 16},
            calibration={"type": "carm", "views": [carm.to_dict()]},
            labels={"procedure": "navigation", "anatomy": "straight"},
        ),
        steps=[
            Step(t=0.0, action={"insertion": 1.0},
                 kinematics={"tip_mm": [0.0, 0.0, 2.0],
                             "node_positions_ref": "000_nodes.npy"},
                 annotations={"device_mask_ref": "000_device_mask.npy",
                              "vessel_mask_ref": "000_vessel_mask.npy",
                              "keypoints": {"tip": {"uv": [8.0, 9.0], "present": True}}},
                 obs_modality="fluoro", obs_ref="000.npy",
                 obs=np.ones((16, 16)),
                 node_positions=np.zeros((3, 3)),
                 annotation_arrays={"device_mask": np.eye(16, dtype=np.uint8),
                                    "vessel_mask": np.ones((16, 16), dtype=np.uint8)}),
        ],
        outcome=Outcome(success=True, final_dist=0.5, steps=1, label="straight_success",
                        metrics={"tip_target": {"success": True}}),
        asset=procedural.straight_tube(80.0, 2.0),
    )
    ep.save(tmp_path / "case")

    out_path = tmp_path / "index.jsonl"
    index_main([str(tmp_path), "--out", str(out_path), "--check-sidecars"])

    msg = capsys.readouterr().out
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert "indexed 1 step records from 1 case bundles" in msg
    assert len(rows) == 1
    row = rows[0]
    assert row["episode"] == "case"
    assert row["episode_dir"] == "case"
    assert row["label"] == "straight_success"
    assert row["obs_modality"] == "fluoro"
    assert row["obs_path"] == "case/obs/000.npy"
    assert row["device_mask_path"] == "case/obs/000_device_mask.npy"
    assert row["vessel_mask_path"] == "case/obs/000_vessel_mask.npy"
    assert row["node_positions_path"] == "case/obs/000_nodes.npy"
    assert row["keypoints"]["tip"]["present"] is True
    assert row["labels"] == {
        "procedure": "navigation",
        "anatomy": "straight",
        "outcome": "straight_success",
    }
    assert row["clinical_metrics"]["tip_target"]["success"] is True
    assert row["calibration_type"] == "carm"

    nested_path = tmp_path / "indexes" / "case" / "index.jsonl"
    index_main([str(tmp_path), "--out", str(nested_path), "--check-sidecars"])
    assert nested_path.exists()

    abs_path = tmp_path / "absolute.jsonl"
    index_main([str(tmp_path), "--out", str(abs_path), "--absolute-paths"])
    abs_row = json.loads(abs_path.read_text().splitlines()[0])
    assert abs_row["obs_path"].endswith("/case/obs/000.npy")
    assert abs_row["obs_path"].startswith("/")

    resolved = next(iter_index_records(out_path))
    assert resolved["obs_path"].endswith("/case/obs/000.npy")
    assert resolved["device_mask_path"].endswith("/case/obs/000_device_mask.npy")

    sample = next(iter_index_records(out_path, load_arrays=True))
    assert sample["obs"].shape == (16, 16)
    assert sample["device_mask"].shape == (16, 16)
    assert sample["vessel_mask"].shape == (16, 16)
    assert sample["node_positions"].shape == (3, 3)

    direct = load_step_record(row, base_dir=tmp_path)
    assert np.array_equal(direct["device_mask"], np.eye(16, dtype=np.uint8))
