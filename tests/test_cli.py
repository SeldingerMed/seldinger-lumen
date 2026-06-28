"""Installed command entry points for first-run workflows."""

import json
import shutil
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
        "lumen-validate": "lumen.cli:validate_main",
        "lumen-index": "lumen.cli:index_main",
        "lumen-inspect-index": "lumen.cli:inspect_index_main",
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


def test_index_inspection_summarizes_and_path_checks_jsonl(tmp_path, capsys):
    from lumen.cli import index_main, inspect_index_main, main

    carm = CArm.looking_at([0.0, 0.0, 40.0], axis=(1.0, 0.0, 0.0), nu=8, nv=8)
    ep = Episode(
        meta=EpisodeMeta(
            asset_ref="asset.json",
            device={"guidewire": {"radius": 0.2}},
            sensor={"modality": "fluoro", "nu": 8, "nv": 8},
            calibration={"type": "carm", "views": [carm.to_dict()]},
            labels={"procedure": "navigation"},
        ),
        steps=[
            Step(t=0.0, action={"insertion": 1.0},
                 kinematics={"tip_mm": [0.0, 0.0, 2.0]},
                 annotations={"device_mask_ref": "000_device_mask.npy",
                              "vessel_mask_ref": "000_vessel_mask.npy",
                              "keypoints": {
                                  "base": {"uv": [4.0, 1.0], "present": True},
                                  "tip": {"uv": [4.0, 3.0], "present": True},
                              }},
                 obs_modality="fluoro", obs_ref="000.npy",
                 obs=np.ones((8, 8)),
                 annotation_arrays={"device_mask": np.eye(8, dtype=np.uint8),
                                    "vessel_mask": np.ones((8, 8), dtype=np.uint8)}),
            Step(t=1.0, action={"insertion": 1.0},
                 kinematics={"tip_mm": [0.0, 0.0, 3.0]},
                 annotations={"device_mask_ref": "001_device_mask.npy",
                              "vessel_mask_ref": "001_vessel_mask.npy",
                              "keypoints": {
                                  "base": {"uv": [4.0, 1.0], "present": True},
                                  "tip": {"uv": [4.0, 4.0], "present": True},
                              }},
                 obs_modality="fluoro", obs_ref="001.npy",
                 obs=np.ones((8, 8)),
                 annotation_arrays={"device_mask": np.eye(8, dtype=np.uint8),
                                    "vessel_mask": np.ones((8, 8), dtype=np.uint8)}),
        ],
        outcome=Outcome(success=True, final_dist=0.5, steps=2, label="inspect_case",
                        metrics={"tip_target": {"success": True, "final_dist": 0.5},
                                 "wall_safety": {"perforation_risk": False}}),
        asset=procedural.straight_tube(80.0, 2.0),
    )
    ep.save(tmp_path / "case")
    index_path = tmp_path / "indexes" / "index.jsonl"
    index_main([str(tmp_path), "--out", str(index_path)])
    capsys.readouterr()

    main(["inspect-index", str(index_path), "--check-arrays", "--require-cv-labels"])
    human = capsys.readouterr().out
    assert "records: 2" in human
    assert "modalities: fluoro=2" in human
    assert "clinical (episodes):" in human
    assert "outcome_success: true=1" in human
    assert "tip_target_success: true=1" in human
    assert "wall_perforation_risk: false=1" in human
    assert "final_dist: mean=0.500 min=0.500 max=0.500 n=1" in human
    assert "keypoint_steps: 2/2" in human
    assert "keypoints: base=2/2, tip=2/2" in human
    assert "cv_labels_required: true" in human
    assert "arrays: checked" in human
    assert "obs_path: 2 refs, 0 missing" in human

    main(["inspect-index", str(index_path), "--check-arrays", "--require-cv-labels", "--json"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["records"] == 2
    assert summary["episodes"] == {"case": 2}
    assert summary["modalities"] == {"fluoro": 2}
    assert summary["labels"] == {"inspect_case": 2}
    assert summary["clinical"]["outcome_success"] == {"true": 1}
    assert summary["clinical"]["tip_target_success"] == {"true": 1}
    assert summary["clinical"]["wall_perforation_risk"] == {"false": 1}
    assert summary["clinical"]["final_dist"]["mean"] == 0.5
    assert summary["clinical"]["final_dist"]["count"] == 1
    assert summary["clinical"]["episode_inconsistencies"] == []
    assert summary["annotations"]["keypoint_steps"] == 2
    assert summary["annotations"]["keypoints_present"] == {"base": 2, "tip": 2}
    assert summary["annotations"]["keypoints_total"] == {"base": 2, "tip": 2}
    assert summary["annotations"]["cv_labels_required"] is True
    assert summary["annotations"]["cv_label_errors"] == []
    assert summary["paths_checked"] is True
    assert summary["arrays_checked"] is True
    assert summary["array_errors"] == []
    assert summary["path_fields"]["obs_path"] == 2
    assert summary["missing_paths"]["obs_path"] == 0

    np.save(tmp_path / "case" / "obs" / "000_device_mask.npy", np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(index_path), "--check-arrays", "--require-cv-labels"])
    assert seen.value.code == 1
    array_out = capsys.readouterr().out
    assert "array errors:" in array_out
    assert "device_mask nonempty" in array_out
    np.save(tmp_path / "case" / "obs" / "000_device_mask.npy", np.eye(8, dtype=np.uint8))

    inconsistent_path = tmp_path / "indexes" / "inconsistent.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    rows[1]["outcome"]["success"] = False
    inconsistent_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(inconsistent_path)])
    assert seen.value.code == 1
    inconsistent_human = capsys.readouterr().out
    assert "endpoint inconsistencies:" in inconsistent_human

    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(inconsistent_path), "--json"])
    assert seen.value.code == 1
    inconsistent = json.loads(capsys.readouterr().out)
    assert inconsistent["clinical"]["episode_inconsistencies"] == [{
        "episode": "case",
        "first_line": 1,
        "line": 2,
    }]

    weak_cv_path = tmp_path / "indexes" / "weak_cv.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    rows[0]["device_mask_path"] = None
    rows[0]["keypoints"].pop("tip")
    weak_cv_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(weak_cv_path), "--require-cv-labels"])
    assert seen.value.code == 1
    weak_out = capsys.readouterr().out
    assert "cv label errors:" in weak_out
    assert "device_mask_path" in weak_out
    assert "keypoints.tip" in weak_out

    (tmp_path / "case" / "obs" / "000.npy").unlink()
    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(index_path), "--check-paths"])
    assert seen.value.code == 1
    broken_out = capsys.readouterr().out
    assert "obs_path: 2 refs, 1 missing" in broken_out
    assert "missing examples:" in broken_out

    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(index_path), "--check-paths", "--json"])
    assert seen.value.code == 1
    broken = json.loads(capsys.readouterr().out)
    assert broken["missing_paths"]["obs_path"] == 1
    assert broken["missing_path_examples"][0]["field"] == "obs_path"


def test_index_inspection_reports_invalid_inputs_without_traceback(tmp_path, capsys):
    from lumen.cli import inspect_index_main

    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(tmp_path / "missing.jsonl")])
    assert seen.value.code == 1
    assert "no index file" in capsys.readouterr().out

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{bad json}\n")
    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(malformed)])
    assert seen.value.code == 1
    out = capsys.readouterr().out
    assert "invalid index" in out
    assert "line 1: invalid JSON" in out

    wrong_shape = tmp_path / "array.jsonl"
    wrong_shape.write_text("[]\n")
    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(wrong_shape)])
    assert seen.value.code == 1
    out = capsys.readouterr().out
    assert "line 1: expected JSON object" in out


def test_benchmark_cli_writes_submission_notes(tmp_path, monkeypatch):
    import lumen.bench as bench
    from lumen.cli import benchmark_main

    seen = {}

    class DummyScorecard:
        suite_version = "lumen-bench/test"
        name = "forward-baseline"
        per_task = [{
            "name": "nav_tube",
            "tier": "easy",
            "safe_success_rate": 1.0,
            "unsafe_success_rate": 0.0,
            "success_rate": 1.0,
            "mean_steps": 3.0,
            "max_pen": 0.0,
        }]
        overall = {
            "safe_success_rate": 1.0,
            "unsafe_success_rate": 0.0,
            "success_rate": 1.0,
            "max_pen": 0.0,
            "mean_return": 42.0,
        }

        def save(self, path):
            with open(path, "w") as f:
                json.dump({"notes": seen["notes"]}, f)

    def fake_evaluate_policy(_policy, name, notes=None):
        seen["name"] = name
        seen["notes"] = notes
        return DummyScorecard()

    monkeypatch.setattr(bench, "evaluate_policy", fake_evaluate_policy)
    monkeypatch.setattr(bench, "leaderboard", lambda _results_dir: [DummyScorecard()])
    monkeypatch.setattr(bench, "scorecard_rejections", lambda _results_dir: [])

    benchmark_main([str(tmp_path)])

    assert seen["name"] == "forward-baseline"
    assert seen["notes"] == {
        "policy": "lumen.bench.forward_policy",
        "command": "lumen benchmark",
        "safety_max_pen": bench.SAFETY_MAX_PEN,
    }
    saved = json.loads((tmp_path / "forward-baseline.json").read_text())
    assert saved["notes"] == seen["notes"]


def test_validate_cli_checks_case_bundles_and_fails_invalid_ones(tmp_path, capsys):
    from lumen.cli import validate_main

    carm = CArm.looking_at([0.0, 0.0, 40.0], axis=(1.0, 0.0, 0.0), nu=16, nv=16)
    ep = Episode(
        meta=EpisodeMeta(
            asset_ref="asset.json",
            device={"guidewire": {"radius": 0.2}},
            sensor={"modality": "fluoro", "nu": 16, "nv": 16},
            calibration={"type": "carm", "views": [carm.to_dict()]},
            labels={"procedure": "navigation"},
        ),
        steps=[
            Step(t=0.0, action={"insertion": 1.0},
                 kinematics={"tip_mm": [0.0, 0.0, 2.0]},
                 annotations={"device_mask_ref": "000_device_mask.npy",
                              "vessel_mask_ref": "000_vessel_mask.npy",
                              "keypoints": {
                                  "base": {"uv": [8.0, 1.0], "present": True},
                                  "tip": {"uv": [8.0, 9.0], "present": True},
                              }},
                 obs_modality="fluoro", obs_ref="000.npy",
                 obs=np.ones((16, 16)),
                 annotation_arrays={"device_mask": np.eye(16, dtype=np.uint8),
                                    "vessel_mask": np.ones((16, 16), dtype=np.uint8)}),
        ],
        outcome=Outcome(success=True, final_dist=0.5, steps=1, label="straight_success"),
        asset=procedural.straight_tube(80.0, 2.0),
    )
    ep.save(tmp_path / "ok")
    validate_main([str(tmp_path)])
    assert "validated 1 case bundles" in capsys.readouterr().out
    validate_main([str(tmp_path), "--require-cv-labels"])
    strict_out = capsys.readouterr().out
    assert "validated 1 case bundles" in strict_out
    assert "cv_label_steps=1" in strict_out

    missing_cv = Episode(
        meta=ep.meta,
        steps=[
            Step(t=0.0, action={"insertion": 1.0},
                 kinematics={"tip_mm": [0.0, 0.0, 2.0]},
                 obs_modality="fluoro", obs_ref="000.npy",
                 obs=np.ones((16, 16))),
        ],
        outcome=ep.outcome,
        asset=procedural.straight_tube(80.0, 2.0),
    )
    missing_cv.save(tmp_path / "missing_cv")
    with pytest.raises(SystemExit) as seen:
        validate_main([str(tmp_path), "--require-cv-labels"])
    assert seen.value.code == 1
    out = capsys.readouterr().out
    assert "missing CV labels" in out
    assert "device_mask_ref" in out

    shutil.rmtree(tmp_path / "missing_cv")
    np.save(tmp_path / "ok" / "obs" / "000_device_mask.npy", np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(SystemExit) as seen:
        validate_main([str(tmp_path), "--require-cv-labels"])
    assert seen.value.code == 1
    out = capsys.readouterr().out
    assert "device_mask nonempty" in out

    (tmp_path / "ok" / "asset.json").unlink()
    with pytest.raises(SystemExit) as seen:
        validate_main([str(tmp_path)])
    assert seen.value.code == 1
    out = capsys.readouterr().out
    assert "invalid bundles" in out
    assert "asset_ref" in out or "asset sidecar" in out


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


def test_index_cli_fails_missing_or_empty_corpus_without_artifact(tmp_path, capsys):
    from lumen.cli import index_main

    missing_out = tmp_path / "missing.jsonl"
    with pytest.raises(SystemExit) as seen:
        index_main([str(tmp_path / "missing"), "--out", str(missing_out)])
    assert seen.value.code == 1
    assert "no episodes under" in capsys.readouterr().out
    assert not missing_out.exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    empty_out = tmp_path / "empty.jsonl"
    with pytest.raises(SystemExit) as seen:
        index_main([str(empty), "--out", str(empty_out)])
    assert seen.value.code == 1
    assert "indexed 0 step records" in capsys.readouterr().out
    assert not empty_out.exists()


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
                              "keypoints": {
                                  "base": {"uv": [8.0, 1.0], "present": True},
                                  "tip": {"uv": [8.0, 9.0], "present": True},
                              }},
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
    nested_sample = next(iter_index_records(nested_path, load_arrays=True))
    assert nested_sample["obs"].shape == (16, 16)
    assert nested_sample["obs_path"] == str((tmp_path / "case" / "obs" / "000.npy").resolve())
    strict_path = tmp_path / "strict.jsonl"
    index_main([str(tmp_path), "--out", str(strict_path), "--require-cv-labels"])
    assert "cv_label_steps=1" in capsys.readouterr().out

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

    (tmp_path / "case" / "obs" / "000_device_mask.npy").unlink()
    with pytest.raises(SystemExit) as seen:
        index_main([str(tmp_path), "--out", str(tmp_path / "bad.jsonl"), "--require-cv-labels"])
    assert seen.value.code == 1
    assert "skipped invalid bundles" in capsys.readouterr().out


def test_index_cli_filters_by_observation_modality(tmp_path, capsys):
    from lumen.cli import index_main

    carm = CArm.looking_at([0.0, 0.0, 40.0], axis=(1.0, 0.0, 0.0), nu=16, nv=16)
    asset = procedural.straight_tube(80.0, 2.0)
    Episode(
        meta=EpisodeMeta(
            asset_ref="asset.json",
            device={"guidewire": {"radius": 0.2}},
            sensor={"modality": "fluoro", "nu": 16, "nv": 16},
            calibration={"type": "carm", "views": [carm.to_dict()]},
            labels={"procedure": "navigation"},
        ),
        steps=[
            Step(t=0.0, action={"insertion": 1.0},
                 kinematics={"tip_mm": [0.0, 0.0, 2.0]},
                 annotations={"device_mask_ref": "000_device_mask.npy",
                              "vessel_mask_ref": "000_vessel_mask.npy",
                              "keypoints": {
                                  "base": {"uv": [8.0, 1.0], "present": True},
                                  "tip": {"uv": [8.0, 9.0], "present": True},
                              }},
                 obs_modality="fluoro", obs_ref="000.npy",
                 obs=np.ones((16, 16)),
                 annotation_arrays={"device_mask": np.eye(16, dtype=np.uint8),
                                    "vessel_mask": np.ones((16, 16), dtype=np.uint8)}),
        ],
        outcome=Outcome(success=True, final_dist=0.5, steps=1, label="fluoro_case"),
        asset=asset,
    ).save(tmp_path / "fluoro")
    Episode(
        meta=EpisodeMeta(
            asset_ref="asset.json",
            device={"scope": {"diameter": 2.0}},
            sensor={"modality": "luminal", "nu": 8, "nv": 8},
            calibration={"type": "scope", "intrinsics": {"fov_deg": 90.0}},
            labels={"procedure": "navigation"},
        ),
        steps=[
            Step(t=0.0, action={"insertion": 1.0},
                 kinematics={"tip_mm": [0.0, 0.0, 2.0]},
                 obs_modality="luminal", obs_ref="000.npy",
                 obs=np.ones((8, 8, 3))),
        ],
        outcome=Outcome(success=True, final_dist=0.5, steps=1, label="luminal_case"),
        asset=asset,
    ).save(tmp_path / "luminal")

    fluoro_index = tmp_path / "fluoro.jsonl"
    index_main([str(tmp_path), "--out", str(fluoro_index),
                "--modality", "fluoro", "--require-cv-labels"])
    fluoro_rows = [json.loads(line) for line in fluoro_index.read_text().splitlines()]
    assert [row["obs_modality"] for row in fluoro_rows] == ["fluoro"]
    out = capsys.readouterr().out
    assert "indexed 1 step records from 1/2 valid case bundles" in out
    assert "modality=fluoro" in out
    assert "cv_label_steps=1" in out

    luminal_index = tmp_path / "luminal.jsonl"
    index_main([str(tmp_path), "--out", str(luminal_index), "--modality", "luminal"])
    luminal_rows = [json.loads(line) for line in luminal_index.read_text().splitlines()]
    assert [row["obs_modality"] for row in luminal_rows] == ["luminal"]
    out = capsys.readouterr().out
    assert "indexed 1 step records from 1/2 valid case bundles" in out
    assert "modality=luminal" in out

    with pytest.raises(SystemExit) as seen:
        index_main([str(tmp_path), "--out", str(tmp_path / "bad.jsonl"),
                    "--modality", "luminal", "--require-cv-labels"])
    assert seen.value.code == 2

    none_index = tmp_path / "none.jsonl"
    with pytest.raises(SystemExit) as seen:
        index_main([str(tmp_path), "--out", str(none_index), "--modality", "none"])
    assert seen.value.code == 1
    out = capsys.readouterr().out
    assert "indexed 0 step records from 0/2 valid case bundles" in out
    assert "no index records emitted" in out
    assert not none_index.exists()
