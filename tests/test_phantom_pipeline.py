"""Frozen-split, actual offline CLI acceptance."""

import json
import subprocess
import sys

import pytest

from lumen.experiments.phantom_capture import admit, synthetic_run, AdmissionError


def test_missing_motion_evidence_is_not_a_measured_trace():
    run = synthetic_run("a", "b")
    run["metadata"]["motion_exit"] = None
    with pytest.raises(AdmissionError, match="motion_exit"):
        admit(run)


def test_demo_uses_real_solver_and_reports_only_matched_holdout(tmp_path):
    pytest.importorskip("newton")
    dest = tmp_path / "demo"
    subprocess.run(
        [sys.executable, "-m", "lumen.experiments.phantom_replay", "demo", str(dest)], check=True
    )
    result = json.loads((dest / "evaluation" / "hold01.json").read_text())
    assert result["run_id"] == "hold01" and result["provenance"] == "synthetic"
    assert result["metrics"]["matched_xy_errors"]["wire"]["count"] == 4
    assert result["metrics"]["matched_xy_errors"]["catheter"]["mean_mm"] is None
    assert result["metrics"]["first_consequential_discrepancy"]["event"] == "matched_xy_error"
    assert result["metrics"]["first_consequential_discrepancy"]["t_s"] == 0.08
    assert len({tuple(r["sim_xy"]["wire"]) for r in result["rows"]}) == 5
    assert result["rows"][2]["measured_xy"]["wire"] is None
    assert all(r["inlet_force_n"] is None for r in result["rows"])
    assert (dest / "evaluation" / "hold01.html").is_file()


def test_positive_absolute_position_can_reverse_before_crossing_zero():
    from lumen.experiments.phantom_metrics import compare

    rows = []
    for i, (base, tip) in enumerate(zip([0, 1, 2, 1, 0], [10, 11, 12, 13, 12])):
        rows.append(
            dict(
                t_s=i * 0.02,
                wire=[base, 0],
                measured_xy={"wire": [0, tip]},
                sim_xy={"wire": [0, 10 + base]},
            )
        )
    rev = compare(rows)["first_reversal"]["wire"]
    assert rev["input_t_s"] == 0.04
    assert rev["measured_lag_s"] == pytest.approx(0.02)
    assert rev["sim_lag_s"] == 0


def test_overlapping_daughters_cannot_identify_a_branch():
    from lumen.experiments.phantom_metrics import compare

    rows = [dict(t_s=0, wire=[0, 0], measured_xy={"wire": [0, 50.5]}, sim_xy={"wire": [0, 50.5]})]
    assert compare(rows)["branch_choice"]["wire"]["sim"] is None


def test_real_solver_projects_distal_endpoint_not_segment_midpoint():
    pytest.importorskip("newton")
    from lumen.experiments.phantom_solver import replay

    out = replay([dict(t_s=0, wire=[0, 0], catheter=[0, 0])], {"wire": [1, 1], "catheter": [1, 1]})
    assert out[0]["wire"][1] == pytest.approx(18, abs=0.001)
    assert out[0]["catheter"][1] == pytest.approx(14, abs=0.001)


def test_export_consumed_as_lumen_data_steps_and_bench_metadata(tmp_path):
    pytest.importorskip("newton")
    from lumen.experiments.phantom_replay import main
    from lumen.data.schema import Step
    from lumen.deployment import BenchTraceMetadata

    dest = tmp_path / "fixture"
    main(["demo", str(dest)])
    r = json.loads((dest / "evaluation/hold01.json").read_text())
    meta = BenchTraceMetadata(**r["bench_trace_metadata"])
    assert meta.torque_units == "not_measured" and meta.source == "bench"
    steps = [Step.from_dict(s) for s in r["data_steps"]]
    assert steps[1].action == {"insertion": 1, "twist": pytest.approx(0.05235987756)}
    assert steps[2].kinematics["measured_xy"]["wire"] is None
    assert steps[2].force is None


def test_split_is_frozen_without_opening_holdout_and_rejects_relabelled_copy(tmp_path):
    pytest.importorskip("newton")
    from lumen.experiments.phantom_replay import calibrate, evaluate, write, METHOD

    write(tmp_path / "plan.json", dict(method=METHOD, calibration=["A"], holdout=["B"]))
    a = synthetic_run("A", "capture-A")
    write(tmp_path / "A.json", a)
    freeze = calibrate(tmp_path)  # B does not exist yet
    a["metadata"].update(run_id="B", recording_id="different-label")
    a["rows"][0]["command"] = {"advance_mm": -99}
    write(tmp_path / "B.json", a)
    with pytest.raises(ValueError, match="duplicate motion"):
        evaluate(tmp_path, freeze)
    assert not (tmp_path / "evaluation").exists()
    with pytest.raises(ValueError, match="frozen fit/plan mismatch"):
        evaluate(tmp_path, "0" * 64)
