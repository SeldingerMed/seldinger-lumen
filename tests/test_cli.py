"""Installed command entry points for first-run workflows."""

import json
import shutil
import subprocess
import sys
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
        "lumen-materialize-batch": "lumen.cli:materialize_batch_main",
        "lumen-split-index": "lumen.cli:split_index_main",
        "lumen-calibrate": "lumen.cli:calibrate_main",
    }


def test_umbrella_cli_dispatches_workflows(capsys):
    from lumen.cli import main

    main(["hardware"])

    payload = json.loads(capsys.readouterr().out)
    assert "newton_available" in payload
    assert "backend_validated" in payload


def test_cli_module_execution_prints_help():
    result = subprocess.run(
        [sys.executable, "-m", "lumen.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "usage: lumen" in result.stdout
    assert "inspect-index" in result.stdout


def test_umbrella_cli_subcommand_help_uses_subcommand_prog(capsys):
    from lumen.cli import main

    with pytest.raises(SystemExit) as seen:
        main(["index", "--help"])

    assert seen.value.code == 0
    out = capsys.readouterr().out
    assert "usage: lumen index" in out
    assert "--check-sidecars" in out


def test_materialize_batch_cli_exports_npz_and_manifest(tmp_path, capsys):
    from lumen.cli import index_main, materialize_batch_main, main
    from lumen.data import materialize_index_batch

    carm = CArm.looking_at([0.0, 0.0, 40.0], axis=(1.0, 0.0, 0.0), nu=8, nv=8)
    steps = []
    for i in range(2):
        steps.append(
            Step(t=float(i), action={"insertion": 1.0 + i, "rotation": 0.1 * i},
                 kinematics={"tip_mm": [0.0, 0.0, 2.0 + i]},
                 annotations={"device_mask_ref": f"{i:03d}_device_mask.npy",
                              "vessel_mask_ref": f"{i:03d}_vessel_mask.npy",
                              "keypoints": {
                                  "base": {"uv": [1.0, 1.0 + i], "present": True},
                                  "tip": {"uv": [4.0, 4.0 + i], "present": True},
                              }},
                 obs_modality="fluoro", obs_ref=f"{i:03d}.npy",
                 obs=np.full((8, 8), float(i)),
                 annotation_arrays={"device_mask": np.eye(8, dtype=np.uint8),
                                    "vessel_mask": np.ones((8, 8), dtype=np.uint8)}),
        )
    Episode(
        meta=EpisodeMeta(
            asset_ref="asset.json",
            device={"guidewire": {"radius": 0.2}},
            sensor={"modality": "fluoro", "nu": 8, "nv": 8},
            calibration={"type": "carm", "views": [carm.to_dict()]},
            labels={"procedure": "navigation"},
        ),
        steps=steps,
        outcome=Outcome(success=True, final_dist=0.5, steps=2, label="batch_case"),
        asset=procedural.straight_tube(80.0, 2.0),
    ).save(tmp_path / "case")
    index_path = tmp_path / "index.jsonl"
    index_main([str(tmp_path), "--out", str(index_path), "--require-cv-labels"])
    capsys.readouterr()

    out_npz = tmp_path / "batch.npz"
    main(["materialize-batch", str(index_path), str(out_npz), "--limit", "2"])
    out = capsys.readouterr().out
    assert "materialized 2 records" in out
    assert "manifest:" in out

    with np.load(out_npz) as batch:
        assert batch["obs"].shape == (2, 8, 8)
        assert batch["device_mask"].dtype == np.uint8
        assert batch["vessel_mask"].shape == (2, 8, 8)
        assert batch["tip_uv"].tolist() == [[4.0, 4.0], [4.0, 5.0]]
        assert batch["base_uv"].tolist() == [[1.0, 1.0], [1.0, 2.0]]
        assert batch["actions"].shape == (2, 2)
        np.testing.assert_allclose(batch["actions"], [[1.0, 0.0], [2.0, 0.1]])
    manifest = json.loads((tmp_path / "batch.npz.manifest.json").read_text())
    assert manifest["records"] == 2
    assert manifest["arrays"]["obs"]["shape"] == [2, 8, 8]
    assert manifest["action_keys"] == ["insertion", "rotation"]
    assert manifest["rows"][0]["labels"]["outcome"] == "batch_case"

    direct_npz = tmp_path / "direct.npz"
    direct = materialize_index_batch(index_path, direct_npz, limit=1, fields=["obs"])
    assert direct["records"] == 1
    with np.load(direct_npz) as batch:
        assert set(batch.files) == {"obs", "actions", "tip_uv", "base_uv"}

    suffixless = tmp_path / "suffixless_batch"
    suffixless_manifest = materialize_index_batch(index_path, suffixless, limit=1, fields=["obs"])
    assert suffixless_manifest["out_npz"] == str(tmp_path / "suffixless_batch.npz")
    assert suffixless_manifest["manifest_path"] == str(tmp_path / "suffixless_batch.npz.manifest.json")
    assert (tmp_path / "suffixless_batch.npz").exists()

    np.save(tmp_path / "case" / "obs" / "001.npy", np.ones((4, 4), dtype=np.float32))
    with pytest.raises(SystemExit) as seen:
        materialize_batch_main([str(index_path), str(tmp_path / "bad.npz")])
    assert seen.value.code == 1
    assert "field 'obs' is not uniform" in capsys.readouterr().out


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
                                  "base": {"uv": [1.0, 1.0], "present": True},
                                  "tip": {"uv": [3.0, 3.0], "present": True},
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
                                  "base": {"uv": [1.0, 1.0], "present": True},
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
    assert "array payloads:" in human
    assert "obs: (8, 8) float64 n=2" in human
    assert "device_mask: (8, 8) uint8 n=2" in human
    assert "vessel_mask: (8, 8) uint8 n=2" in human
    assert "keypoint_mask_tolerance: 1.500px" in human
    assert "mask coverage:" in human
    assert "device_mask: mean=12.500% min=12.500% max=12.500% n=2" in human
    assert "vessel_mask: mean=100.000% min=100.000% max=100.000% n=2" in human
    assert "keypoint device distance:" in human
    assert "base: mean=0.000px min=0.000px max=0.000px n=2" in human
    assert "tip: mean=0.000px min=0.000px max=0.000px n=2" in human
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
    assert summary["annotations"]["keypoint_errors"] == []
    assert summary["annotations"]["keypoint_mask_tolerance_px"] == 1.5
    assert summary["paths_checked"] is True
    assert summary["arrays_checked"] is True
    assert summary["array_errors"] == []
    assert summary["array_payloads"]["obs"] == [{
        "shape": [8, 8],
        "dtype": "float64",
        "count": 2,
    }]
    assert summary["array_payloads"]["device_mask"] == [{
        "shape": [8, 8],
        "dtype": "uint8",
        "count": 2,
    }]
    assert summary["array_payloads"]["vessel_mask"] == [{
        "shape": [8, 8],
        "dtype": "uint8",
        "count": 2,
    }]
    assert summary["mask_coverage"]["device_mask"] == {
        "count": 2,
        "min": 0.125,
        "max": 0.125,
        "mean": 0.125,
    }
    assert summary["mask_coverage"]["vessel_mask"] == {
        "count": 2,
        "min": 1.0,
        "max": 1.0,
        "mean": 1.0,
    }
    assert summary["keypoint_device_distance"]["base"] == {
        "count": 2,
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
    }
    assert summary["keypoint_device_distance"]["tip"] == {
        "count": 2,
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
    }
    assert summary["path_fields"]["obs_path"] == 2
    assert summary["missing_paths"]["obs_path"] == 0

    np.save(tmp_path / "case" / "obs" / "000.npy", np.ones((4, 4), dtype=np.float32))
    np.save(tmp_path / "case" / "obs" / "000_device_mask.npy", np.eye(4, dtype=np.uint8))
    np.save(tmp_path / "case" / "obs" / "000_vessel_mask.npy", np.ones((4, 4), dtype=np.uint8))
    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(index_path), "--require-uniform-arrays"])
    assert seen.value.code == 1
    mixed_payload_out = capsys.readouterr().out
    assert "array payload errors:" in mixed_payload_out
    assert "obs" in mixed_payload_out
    assert "(4, 4) float32 n=1" in mixed_payload_out
    assert "(8, 8) float64 n=1" in mixed_payload_out
    np.save(tmp_path / "case" / "obs" / "000.npy", np.ones((8, 8)))
    np.save(tmp_path / "case" / "obs" / "000_device_mask.npy", np.eye(8, dtype=np.uint8))
    np.save(tmp_path / "case" / "obs" / "000_vessel_mask.npy", np.ones((8, 8), dtype=np.uint8))

    np.save(tmp_path / "case" / "obs" / "000_device_mask.npy", np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(index_path), "--check-arrays", "--require-cv-labels"])
    assert seen.value.code == 1
    array_out = capsys.readouterr().out
    assert "array errors:" in array_out
    assert "device_mask nonempty" in array_out
    np.save(tmp_path / "case" / "obs" / "000_device_mask.npy", np.eye(8, dtype=np.uint8))

    bad_keypoint_path = tmp_path / "indexes" / "bad_keypoint.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    rows[0]["keypoints"]["tip"]["uv"] = [12.0, -1.0]
    bad_keypoint_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(bad_keypoint_path), "--check-arrays", "--require-cv-labels"])
    assert seen.value.code == 1
    keypoint_out = capsys.readouterr().out
    assert "keypoint errors:" in keypoint_out
    assert "keypoints.tip in-frame" in keypoint_out

    off_device_path = tmp_path / "indexes" / "off_device_keypoint.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    rows[0]["keypoints"]["tip"]["uv"] = [0.0, 7.0]
    off_device_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(off_device_path), "--check-arrays", "--require-cv-labels"])
    assert seen.value.code == 1
    off_device_out = capsys.readouterr().out
    assert "keypoints.tip on-device" in off_device_out
    inspect_index_main([str(off_device_path), "--check-arrays", "--require-cv-labels",
                        "--keypoint-mask-tolerance", "100"])
    assert "keypoint errors:" not in capsys.readouterr().out

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

    with pytest.raises(SystemExit) as seen:
        inspect_index_main([str(wrong_shape), "--keypoint-mask-tolerance", "-1"])
    assert seen.value.code == 2
    assert "--keypoint-mask-tolerance must be non-negative" in capsys.readouterr().err


def test_index_record_iterator_reports_invalid_rows_with_context(tmp_path):
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("\n{bad json}\n")
    with pytest.raises(ValueError, match=r"malformed\.jsonl: line 2: invalid JSON"):
        list(iter_index_records(malformed))

    wrong_shape = tmp_path / "array.jsonl"
    wrong_shape.write_text("[]\n")
    with pytest.raises(ValueError, match=r"array\.jsonl: line 1: expected JSON object, got list"):
        list(iter_index_records(wrong_shape))


def test_device_keypoint_mask_error_helper_is_exported():
    from lumen.data import device_keypoint_mask_distances, device_keypoint_mask_errors

    mask = np.eye(8, dtype=np.uint8)

    errors = device_keypoint_mask_errors(
        {"tip": {"uv": [7.0, 0.0], "present": True}},
        mask,
    )
    assert len(errors) == 1
    assert errors[0].startswith("keypoints.tip on-device distance=")
    assert device_keypoint_mask_errors(
        {"tip": {"uv": [7.0, 0.0], "present": True}},
        mask,
        mask_tolerance_px=10.0,
    ) == []
    distances = device_keypoint_mask_distances(
        {"tip": {"uv": [7.0, 0.0], "present": True}},
        mask,
    )
    assert set(distances) == {"tip"}
    assert len(distances["tip"]) == 1
    assert distances["tip"][0] > 0.0


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
                                  "base": {"uv": [1.0, 1.0], "present": True},
                                  "tip": {"uv": [9.0, 9.0], "present": True},
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

    manifest = tmp_path / "ok" / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["steps"][0]["annotations"]["keypoints"]["tip"]["uv"] = [0.0, 15.0]
    manifest.write_text(json.dumps(payload) + "\n")
    with pytest.raises(SystemExit) as seen:
        validate_main([str(tmp_path), "--require-cv-labels"])
    assert seen.value.code == 1
    out = capsys.readouterr().out
    assert "keypoints.tip on-device" in out
    validate_main([str(tmp_path), "--require-cv-labels", "--keypoint-mask-tolerance", "100"])
    assert "validated 1 case bundles" in capsys.readouterr().out
    payload["steps"][0]["annotations"]["keypoints"]["tip"]["uv"] = [9.0, 9.0]
    manifest.write_text(json.dumps(payload) + "\n")

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
    assert "run `lumen capture" in out
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
                                  "base": {"uv": [1.0, 1.0], "present": True},
                                  "tip": {"uv": [9.0, 9.0], "present": True},
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

    manifest = tmp_path / "case" / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["steps"][0]["annotations"]["keypoints"]["tip"]["uv"] = [0.0, 15.0]
    manifest.write_text(json.dumps(payload) + "\n")
    with pytest.raises(SystemExit) as seen:
        index_main([str(tmp_path), "--out", str(tmp_path / "off_device.jsonl"),
                    "--require-cv-labels"])
    assert seen.value.code == 1
    assert "keypoints.tip on-device" in capsys.readouterr().out
    assert not (tmp_path / "off_device.jsonl").exists()
    index_main([str(tmp_path), "--out", str(tmp_path / "loose.jsonl"),
                "--require-cv-labels", "--keypoint-mask-tolerance", "100"])
    assert "cv_label_steps=1" in capsys.readouterr().out
    payload["steps"][0]["annotations"]["keypoints"]["tip"]["uv"] = [9.0, 9.0]
    manifest.write_text(json.dumps(payload) + "\n")

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
    with pytest.raises(FileNotFoundError, match="missing device_mask_path.*episode 'case' step 0"):
        next(iter_index_records(out_path, load_arrays=True))
    with pytest.raises(SystemExit) as seen:
        index_main([str(tmp_path), "--out", str(tmp_path / "bad.jsonl"), "--require-cv-labels"])
    assert seen.value.code == 1
    out = capsys.readouterr().out
    assert "index failed before writing" in out
    assert "skipped invalid bundles" in out
    assert not (tmp_path / "bad.jsonl").exists()


def test_load_fluoro_index_example_stacks_training_batch(tmp_path, capsys):
    from examples.load_fluoro_index import load_batch, main
    from lumen.cli import index_main

    carm = CArm.looking_at([0.0, 0.0, 40.0], axis=(1.0, 0.0, 0.0), nu=8, nv=8)
    steps = []
    for i in range(2):
        steps.append(
            Step(t=float(i), action={"insertion": 1.0},
                 kinematics={"tip_mm": [0.0, 0.0, 2.0 + i]},
                 annotations={"device_mask_ref": f"{i:03d}_device_mask.npy",
                              "vessel_mask_ref": f"{i:03d}_vessel_mask.npy",
                              "keypoints": {
                                  "base": {"uv": [1.0, 1.0 + i], "present": True},
                                  "tip": {"uv": [4.0, 4.0 + i], "present": True},
                              }},
                 obs_modality="fluoro", obs_ref=f"{i:03d}.npy",
                 obs=np.full((8, 8), float(i)),
                 annotation_arrays={"device_mask": np.eye(8, dtype=np.uint8),
                                    "vessel_mask": np.ones((8, 8), dtype=np.uint8)}),
        )
    Episode(
        meta=EpisodeMeta(
            asset_ref="asset.json",
            device={"guidewire": {"radius": 0.2}},
            sensor={"modality": "fluoro", "nu": 8, "nv": 8},
            calibration={"type": "carm", "views": [carm.to_dict()]},
            labels={"procedure": "navigation"},
        ),
        steps=steps,
        outcome=Outcome(success=True, final_dist=0.5, steps=2, label="training_case"),
        asset=procedural.straight_tube(80.0, 2.0),
    ).save(tmp_path / "case")
    index_path = tmp_path / "fluoro.jsonl"
    index_main([str(tmp_path), "--out", str(index_path),
                "--modality", "fluoro", "--require-cv-labels"])
    capsys.readouterr()

    batch = load_batch(index_path, limit=2)

    assert batch["obs"].shape == (2, 8, 8)
    assert batch["device_mask"].dtype == np.uint8
    assert batch["vessel_mask"].shape == (2, 8, 8)
    assert batch["tip_uv"].tolist() == [[4.0, 4.0], [4.0, 5.0]]
    assert batch["base_uv"].tolist() == [[1.0, 1.0], [1.0, 2.0]]
    assert batch["labels"] == ["training_case", "training_case"]

    main(str(index_path), limit=2)
    out = capsys.readouterr().out
    assert "obs: (2, 8, 8) float64" in out
    assert "tip_uv: (2, 2) float64" in out

    np.save(tmp_path / "case" / "obs" / "000.npy", np.ones((4, 4), dtype=np.float32))
    np.save(tmp_path / "case" / "obs" / "000_device_mask.npy", np.eye(4, dtype=np.uint8))
    np.save(tmp_path / "case" / "obs" / "000_vessel_mask.npy", np.ones((4, 4), dtype=np.uint8))
    result = subprocess.run(
        [sys.executable, "examples/load_fluoro_index.py", str(index_path), "--limit", "2"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "require-uniform-arrays" in result.stderr


def test_index_split_cli_writes_episode_grouped_manifests(tmp_path, capsys):
    from lumen.cli import index_main, split_index_main, main
    from lumen.data import split_index_records

    carm = CArm.looking_at([0.0, 0.0, 40.0], axis=(1.0, 0.0, 0.0), nu=8, nv=8)
    asset = procedural.straight_tube(80.0, 2.0)
    for i, (name, label) in enumerate([
        ("straight_a", "straight_success"),
        ("straight_b", "straight_success"),
        ("branch_a", "branch_success"),
        ("branch_b", "branch_success"),
    ]):
        steps = []
        for j in range(2):
            steps.append(
                Step(t=float(j), action={"insertion": 1.0},
                     kinematics={"tip_mm": [0.0, 0.0, 2.0 + j]},
                     annotations={"device_mask_ref": f"{j:03d}_device_mask.npy",
                                  "vessel_mask_ref": f"{j:03d}_vessel_mask.npy",
                                  "keypoints": {
                                      "base": {"uv": [1.0, 1.0], "present": True},
                                      "tip": {"uv": [4.0, 4.0], "present": True},
                                  }},
                     obs_modality="fluoro", obs_ref=f"{j:03d}.npy",
                     obs=np.full((8, 8), i + j),
                     annotation_arrays={"device_mask": np.eye(8, dtype=np.uint8),
                                        "vessel_mask": np.ones((8, 8), dtype=np.uint8)}),
            )
        Episode(
            meta=EpisodeMeta(
                asset_ref="asset.json",
                device={"guidewire": {"radius": 0.2}},
                sensor={"modality": "fluoro", "nu": 8, "nv": 8},
                calibration={"type": "carm", "views": [carm.to_dict()]},
                labels={"procedure": "navigation", "fold_family": label.split("_")[0]},
            ),
            steps=steps,
            outcome=Outcome(success=True, final_dist=0.5, steps=2, label=label),
            asset=asset,
        ).save(tmp_path / name)

    index_path = tmp_path / "index.jsonl"
    index_main([str(tmp_path), "--out", str(index_path), "--require-cv-labels"])
    capsys.readouterr()

    split_dir = tmp_path / "splits"
    split_index_main([str(index_path), "--out-dir", str(split_dir), "--ratios", "0.5", "0.25", "0.25",
                      "--seed", "7", "--stratify", "label", "obs_modality"])
    out = capsys.readouterr().out
    assert "split 8 records from 4 episodes" in out
    assert "train.jsonl" in out

    manifest = json.loads((split_dir / "manifest.json").read_text())
    assert manifest["source_index"] == str(index_path)
    assert manifest["group_by"] == "episode"
    assert manifest["stratify"] == ["label", "obs_modality"]
    assert manifest["ratios"] == {"train": 0.5, "val": 0.25, "test": 0.25}
    assert manifest["splits"]["train"]["episodes"] == 2
    assert manifest["splits"]["val"]["episodes"] == 1
    assert manifest["splits"]["test"]["episodes"] == 1

    split_rows = {}
    episode_to_split = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}.jsonl"
        assert path.exists()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        split_rows[split] = rows
        for row in rows:
            previous = episode_to_split.setdefault(row["episode"], split)
            assert previous == split
    assert sum(len(rows) for rows in split_rows.values()) == 8
    assert set(episode_to_split) == {"straight_a", "straight_b", "branch_a", "branch_b"}

    repeat_dir = tmp_path / "repeat"
    again = split_index_records(index_path, repeat_dir, ratios=(0.5, 0.25, 0.25), seed=7,
                                stratify_fields=("label", "obs_modality"))
    assert again["assignments"] == manifest["assignments"]

    main(["split-index", str(index_path), "--out-dir", str(tmp_path / "umbrella")])
    assert (tmp_path / "umbrella" / "manifest.json").exists()

    for split in ("train", "val", "test"):
        split_path = split_dir / f"{split}.jsonl"
        loaded = list(iter_index_records(split_path, load_arrays=True))
        assert len(loaded) > 0
        for record in loaded:
            assert record["obs"].shape == (8, 8)
            assert record["device_mask"].shape == (8, 8)
            assert record["vessel_mask"].shape == (8, 8)


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
                                  "base": {"uv": [1.0, 1.0], "present": True},
                                  "tip": {"uv": [9.0, 9.0], "present": True},
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

    shutil.copytree(tmp_path / "fluoro", tmp_path / "z_bad_fluoro")
    manifest = tmp_path / "z_bad_fluoro" / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["steps"][0]["obs_ref"] = "missing.npy"
    manifest.write_text(json.dumps(payload) + "\n")
    strict_path = tmp_path / "strict_check_sidecars.jsonl"
    with pytest.raises(SystemExit) as seen:
        index_main([str(tmp_path), "--out", str(strict_path),
                    "--modality", "fluoro", "--check-sidecars"])
    assert seen.value.code == 1
    out = capsys.readouterr().out
    assert "index failed before writing" in out
    assert "candidate step records" in out
    assert "indexed 1 step records" not in out
    assert "skipped invalid bundles" in out
    assert not strict_path.exists()

    payload["steps"][0]["obs_ref"] = "000.npy"
    payload["steps"][0]["annotations"]["keypoints"]["tip"]["uv"] = [0.0, 15.0]
    manifest.write_text(json.dumps(payload) + "\n")
    with pytest.raises(SystemExit) as seen:
        index_main([str(tmp_path), "--modality", "fluoro", "--require-cv-labels"])
    assert seen.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "keypoints.tip on-device" in captured.err
    shutil.rmtree(tmp_path / "z_bad_fluoro")

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
