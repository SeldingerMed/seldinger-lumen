"""Tests for strict normalized capture-export admission (lumen.experiments.phantom_capture).

Contract under test:
  admit(run) -> list of normalized rows (one per timestamp), or AdmissionError.
  A run is a dict with:
    - metadata: run_id, recording_id, phantom_id, provenance ('synthetic' or
      'authorized_phantom'), consent (True), tool_ids (dict, must include a
      'wire' tool id), calibration_id, homography (non-singular 3x3 pixel ->
      Lumen fork-plane XY mm), clock_sync (non-empty 'evidence' and
      'reference'), signs {'wire': [+-1, +-1], 'catheter': [+-1, +-1]}
      (validated only — applied downstream by the solver), motion_exit (None
      or dict with independent external_tracker/tool_exit evidence),
      canonical units metadata, inlet sensor site/calibration when any row
      carries a non-null inlet_force_n.
    - rows: 1..3001 entries, each a dict with:
        t_s            (finite float, strictly increasing, >= 0)
        wire           [abs mm, unwrapped degrees] or None
        catheter       same, or None (null when not visible)
        command        (logged only; any JSON-able value or None — never evidence)
        measured_xy    {'wire': [x, y] mm or None, 'catheter': [x, y] mm or None}
        inlet_force_n  finite float (N, explicit inlet axial site) or None
        visibility     raw per-tip visibility flags (preserved)
        confidence     raw per-tip confidence (preserved; validated 0..1 or None)
        raw_pixel      optional per-tip raw pixel coords (preserved)
  Projective consistency: a non-null raw_pixel must map through the
  homography onto the paired measured_xy. measured_xy without raw_pixel is
  admitted only when the metadata documents the calibrated-XY source.
  Invisible tip => measured_xy and raw_pixel must be null.
  Negative raw insertion is valid when the polarity sign is -1 (the solver
  checks the canonical sign); admit preserves raw values.
"""

import math

import pytest

from lumen.experiments.phantom_capture import (
    AdmissionError,
    MAX_RUN_LENGTH,
    admit,
    synthetic_run,
)

# ---------------------------------------------------------------- fixtures


def _good_run():
    return synthetic_run("run-1", "rec-1", variant="straight")


# ------------------------------------------------- metadata validation


def test_valid_synthetic_run_admits():
    rows = admit(_good_run())
    assert 1 <= len(rows) <= MAX_RUN_LENGTH
    for r in rows:
        assert isinstance(r["t_s"], float)
        assert math.isfinite(r["t_s"])
        assert set(r) >= {
            "t_s",
            "wire",
            "catheter",
            "command",
            "measured_xy",
            "inlet_force_n",
            "visibility",
            "confidence",
            "raw_pixel",
        }
        # canonical units live in the KEY names; raw magnitudes preserved
        if r["wire"] is not None:
            assert isinstance(r["wire"][0], float) and isinstance(r["wire"][1], float)
            assert -360.0 <= r["wire"][1] <= 360.0


def test_missing_metadata_rejected():
    for key in [
        "run_id",
        "recording_id",
        "phantom_id",
        "provenance",
        "consent",
        "tool_ids",
        "calibration_id",
        "homography",
        "clock_sync",
        "signs",
        "units",
    ]:
        run = _good_run()
        del run["metadata"][key]
        with pytest.raises(AdmissionError):
            admit(run)


def test_bad_provenance_rejected():
    run = _good_run()
    run["metadata"]["provenance"] = "clinical_leftover"
    with pytest.raises(AdmissionError):
        admit(run)


def test_missing_consent_rejected():
    run = _good_run()
    run["metadata"]["consent"] = False
    with pytest.raises(AdmissionError):
        admit(run)


def test_tool_ids_must_include_wire_tool_id():
    run = _good_run()
    del run["metadata"]["tool_ids"]["wire"]
    with pytest.raises(AdmissionError):
        admit(run)
    run = _good_run()
    run["metadata"]["tool_ids"]["wire"] = ""
    with pytest.raises(AdmissionError):
        admit(run)


def test_clock_sync_requires_evidence_and_reference():
    run = _good_run()
    del run["metadata"]["clock_sync"]["reference"]
    with pytest.raises(AdmissionError):
        admit(run)
    run = _good_run()
    run["metadata"]["clock_sync"]["evidence"] = ""
    with pytest.raises(AdmissionError):
        admit(run)


def test_signs_must_be_pm1_pairs():
    run = _good_run()
    run["metadata"]["signs"]["wire"] = [1, 2]
    with pytest.raises(AdmissionError):
        admit(run)


def test_signs_present_but_not_applied_by_admit():
    run = _good_run()
    run["metadata"]["signs"] = {"wire": [-1, -1], "catheter": [-1, 1]}
    rows = admit(run)
    assert rows[2]["wire"][0] == 4.0  # raw value preserved verbatim


def test_singular_homography_rejected():
    run = _good_run()
    run["metadata"]["homography"] = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]
    with pytest.raises(AdmissionError):
        admit(run)


def test_missing_homography_row_rejected():
    run = _good_run()
    run["metadata"]["homography"] = [1, 2, 3]
    with pytest.raises(AdmissionError):
        admit(run)


def test_canonical_units_metadata_required():
    run = _good_run()
    run["metadata"]["units"] = {"t_s": "ms", "measured_xy": "mm", "inlet_force_n": "N"}
    with pytest.raises(AdmissionError):
        admit(run)


def test_motion_exit_requires_independent_external_evidence():
    run = _good_run()
    run["metadata"]["motion_exit"] = {"source": "external_tracker", "evidence": "rec-2"}
    assert admit(run)
    run = _good_run()
    run["metadata"]["motion_exit"] = {"source": "tool_exit", "evidence": "exit-log-7"}
    assert admit(run)
    # motor command log substituted for independent measurement -> reject
    run = _good_run()
    run["metadata"]["motion_exit"] = {"source": "motor_command", "evidence": "cmd-log"}
    with pytest.raises(AdmissionError):
        admit(run)
    run = _good_run()
    run["metadata"]["motion_exit"] = {"source": "external_tracker"}
    with pytest.raises(AdmissionError):
        admit(run)  # evidence required
    run = _good_run()
    run["metadata"]["motion_exit"] = 42
    with pytest.raises(AdmissionError):
        admit(run)


def test_inlet_force_requires_inlet_sensor_site_and_calibration():
    # straight variant carries non-null inlet force -> sensor metadata required
    run = _good_run()
    assert all(r["inlet_force_n"] is not None for r in run["rows"])
    del run["metadata"]["inlet_sensor"]
    with pytest.raises(AdmissionError):
        admit(run)
    run = _good_run()
    run["metadata"]["inlet_sensor"]["calibration_id"] = ""
    with pytest.raises(AdmissionError):
        admit(run)
    run = _good_run()
    run["metadata"]["inlet_sensor"]["site"] = "torque_z"  # wrong site
    with pytest.raises(AdmissionError):
        admit(run)
    # partial variant: no inlet force -> no sensor metadata required
    partial = synthetic_run("r", "rec", variant="partial")
    assert all(r["inlet_force_n"] is None for r in partial["rows"])
    assert admit(partial)


# ------------------------------------------------- row validation


def test_partial_measurement_rejected():
    run = _good_run()
    run["rows"][2]["measured_xy"] = {"wire": [10.0, None], "catheter": None}
    with pytest.raises(AdmissionError):
        admit(run)


def test_nan_and_inf_rejected():
    for bad in (float("nan"), float("inf")):
        run = _good_run()
        run["rows"][1]["wire"] = [bad, 0.0]
        with pytest.raises(AdmissionError):
            admit(run)
        run = _good_run()
        run["rows"][1]["t_s"] = bad
        with pytest.raises(AdmissionError):
            admit(run)


def test_wrong_types_rejected():
    run = _good_run()
    run["rows"][0]["t_s"] = "0.1"
    with pytest.raises(AdmissionError):
        admit(run)
    run = _good_run()
    run["rows"][0]["wire"] = [1.0, 2.0, 3.0]
    with pytest.raises(AdmissionError):
        admit(run)


def test_negative_abs_mm_with_positive_polarity_rejected():
    run = _good_run()
    run["rows"][0]["wire"] = [-1.0, 5.0]
    with pytest.raises(AdmissionError):
        admit(run)


def test_negative_abs_mm_preserved_with_negative_polarity():
    # solver checks the canonical sign; admit only preserves raw values
    run = _good_run()
    run["metadata"]["signs"] = {"wire": [-1, 1], "catheter": [1, 1]}
    run["rows"][0]["wire"] = [-3.5, 5.0]
    rows = admit(run)
    assert rows[0]["wire"] == [-3.5, 5.0]


def test_non_monotonic_clock_rejected():
    run = _good_run()
    run["rows"][2]["t_s"] = run["rows"][1]["t_s"]
    with pytest.raises(AdmissionError):
        admit(run)


def test_run_length_bounded():
    run = _good_run()
    run["rows"] = list(run["rows"]) + [
        dict(run["rows"][-1], t_s=run["rows"][-1]["t_s"] + 0.1) for _ in range(MAX_RUN_LENGTH)
    ]
    assert len(run["rows"]) > MAX_RUN_LENGTH
    with pytest.raises(AdmissionError):
        admit(run)


def test_inlet_force_must_be_explicit_or_null():
    run = _good_run()
    run["rows"][0]["inlet_force_n"] = "unknown"
    with pytest.raises(AdmissionError):
        admit(run)


def test_confidence_validated_per_tip():
    run = _good_run()
    run["rows"][0]["confidence"] = {"wire": 1.5, "catheter": 0.8}
    with pytest.raises(AdmissionError):
        admit(run)
    run = _good_run()
    run["rows"][0]["confidence"] = {"wire": 0.95}  # missing catheter tip
    with pytest.raises(AdmissionError):
        admit(run)
    run = _good_run()
    run["rows"][0]["confidence"] = {"wire": None, "catheter": 0.8}
    assert admit(run)


def test_visibility_flags_validated_per_tip():
    run = _good_run()
    run["rows"][0]["visibility"] = {"wire": "yes", "catheter": True}
    with pytest.raises(AdmissionError):
        admit(run)


# ------------------------------------------------- admission semantics


def test_invisible_tip_xy_and_raw_pixel_must_be_null():
    run = _good_run()
    row = run["rows"][0]
    row["catheter"] = None
    row["measured_xy"]["catheter"] = None
    row["visibility"]["catheter"] = False
    row["raw_pixel"]["catheter"] = [150.0, 210.0]  # invisible yet has pixels
    with pytest.raises(AdmissionError):
        admit(run)
    row["raw_pixel"]["catheter"] = None
    assert admit(run)[0]["catheter"] is None


def test_raw_pixel_must_project_onto_measured_xy():
    run = _good_run()
    row = run["rows"][0]
    # identity homography: raw_pixel must equal measured_xy
    row["measured_xy"]["wire"] = [1.0, 2.0]
    row["raw_pixel"]["wire"] = [9.0, 9.0]
    with pytest.raises(AdmissionError):
        admit(run)
    row["raw_pixel"]["wire"] = [1.0, 2.0]
    assert admit(run)


def test_projective_map_consistency_nontrivial_homography():
    run = synthetic_run("run-1", "rec-1", variant="straight", n=1)
    # H: pixel (u, v) -> (2u + 1, v)
    run["metadata"]["homography"] = [[2.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    row = run["rows"][0]
    run["rows"][0]["measured_xy"]["catheter"] = [
        301.0,
        210.0,
    ]  # catheter tip consistent under new H
    run["rows"][0]["raw_pixel"]["catheter"] = [150.0, 210.0]
    row["measured_xy"]["wire"] = [21.0, 2.0]
    row["raw_pixel"]["wire"] = [10.0, 2.0]
    rows = admit(run)
    assert rows[0]["measured_xy"]["wire"] == [21.0, 2.0]
    row["measured_xy"]["wire"] = [20.0, 2.0]  # off by the +1 offset
    with pytest.raises(AdmissionError):
        admit(run)


def test_calibrated_xy_without_raw_pixel_requires_documentation():
    run = _good_run()
    run["rows"][0]["raw_pixel"]["wire"] = None  # XY present, pixels absent
    with pytest.raises(AdmissionError):
        admit(run)
    run["metadata"]["measured_xy_source"] = "fluoroscopy-tracked-tip-cal-2026-11"
    assert admit(run)


def test_track_visible_without_motion_instrument():
    # stationary catheter: tracked in images but motion instrument absent
    run = _good_run()
    run["rows"][0]["catheter"] = [12.0, 0.0]
    run["rows"][0]["measured_xy"]["catheter"] = [150.0, 210.0]
    run["rows"][0]["raw_pixel"]["catheter"] = [150.0, 210.0]
    rows = admit(run)
    assert rows[0]["catheter"] == [12.0, 0.0]


def test_raw_visibility_confidence_preserved():
    run = _good_run()
    run["rows"][0]["visibility"] = {"wire": True, "catheter": False}
    run["rows"][0]["measured_xy"]["catheter"] = None
    run["rows"][0]["raw_pixel"]["catheter"] = None
    run["rows"][0]["confidence"] = {"wire": 0.9, "catheter": 0.1}
    rows = admit(run)
    assert rows[0]["visibility"] == {"wire": True, "catheter": False}
    assert rows[0]["confidence"] == {"wire": 0.9, "catheter": 0.1}


def test_command_is_logged_only_and_never_evidence():
    run = _good_run()
    # a garbage motor command must not affect admission (no substitution)
    run["rows"][0]["command"] = {"advance_mm": 999.0, "bogus": [1, 2, 3]}
    rows = admit(run)
    assert rows[0]["command"] == {"advance_mm": 999.0, "bogus": [1, 2, 3]}


# ------------------------------------------------- example factory


def test_synthetic_run_factory_is_self_admitting():
    for variant in ("straight", "partial"):
        rows = admit(synthetic_run("r", "rec", variant=variant, n=5))
        assert 1 <= len(rows) <= MAX_RUN_LENGTH


def test_synthetic_run_rows_are_replayable_20ms_zero_start():
    for variant in ("straight", "partial"):
        rows = admit(synthetic_run("r", "rec", variant=variant, n=5))
        assert [r["t_s"] for r in rows] == [0.0, 0.02, 0.04, 0.06, 0.08]


def test_synthetic_run_straight_variant_coaxial():
    rows = admit(synthetic_run("r", "rec", variant="straight", n=5))
    for r in rows:
        # coaxial: catheter shares the wire tip axis
        assert r["measured_xy"]["catheter"] == r["measured_xy"]["wire"]
        assert r["catheter"] is not None


def test_synthetic_run_partial_variant_wire_only_no_inlet():
    rows = admit(synthetic_run("r", "rec", variant="partial", n=5))
    for r in rows:
        assert r["catheter"] is None
        assert r["measured_xy"]["catheter"] is None
        assert r["inlet_force_n"] is None


def test_synthetic_run_partial_variant_includes_visible_missing_frame():
    run = synthetic_run("r", "rec", variant="partial", n=5)
    rows = admit(run)
    missing = [r for r in rows if r["measured_xy"]["wire"] is None]
    assert all(r["wire"] is not None for r in rows)
    assert len(missing) == 1
    m = missing[0]
    assert m["measured_xy"]["wire"] is None
    assert m["visibility"]["wire"] is True  # flag says visible, frame missing


def test_synthetic_run_variants_trajectory_independent():
    a = admit(synthetic_run("r1", "rec", variant="straight", n=5))
    b = admit(synthetic_run("r2", "rec", variant="partial", n=5))
    a_adv = [r["wire"][0] for r in a if r["wire"] is not None]
    b_adv = [r["wire"][0] for r in b if r["wire"] is not None]
    assert a_adv != b_adv
    assert [r["measured_xy"]["wire"] for r in a if r["measured_xy"]["wire"]] != [
        r["measured_xy"]["wire"] for r in b if r["measured_xy"]["wire"]
    ]


def test_synthetic_run_partial_heldout_discrepancy_on_last_sample():
    rows = admit(synthetic_run("r", "rec", variant="partial", n=5))
    xs = [r["measured_xy"]["wire"] for r in rows if r["measured_xy"]["wire"] is not None]
    assert len(xs) == 4  # one missing frame
    # last injected sample deviates from the analytic trajectory by > 5 mm
    # longitudinally
    assert abs(xs[-1][1] - xs[-2][1]) > 5.0
