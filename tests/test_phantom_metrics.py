"""TDD unit tests for lumen.experiments.phantom_metrics (phantom compare layer).

Row contract (consumed here, produced by the parent instrumentation layer):

    row = {
        "t_s": float,
        "wire":     [signed_insertion_mm, rotation_deg]  | None,
        "catheter": [signed_insertion_mm, rotation_deg]  | None,
        "measured_xy": {"wire": [x, y] | None, "catheter": [x, y] | None},
        "sim_xy":      {"wire": [x, y] | None, "catheter": [x, y] | None},
        "inlet_force_n": float | None,   # optional key
    }

XY is the calibrated fork plane (Lumen X, Z). Fixed procedural bifurcation:
apex at (0, 50) mm, daughter branches at +/-35 deg from the trunk (+Y) axis,
finite segment corridor of radius BRANCH_RADIUS_MM and length
BRANCH_SEGMENT_LENGTH_MM. Per-instrument analysis throughout: each instrument
is scored on its OWN tip and insertion, never mixed. Branch choice reports
``observed_at_s`` (first visible branch sample) independently of the possibly
censored entry time ``t_s``; first observed signs are compared even across
gaps, timestamped at the later observation time. ``branch_entry`` is
per-instrument with a nullable ``signed_lag_s`` and a ``matched_rows``
coverage count. Conservative timing: no inference across sampling gaps; an
occlusion before the first event nulls the event timing. Signed lags
(sim - measured) are reported only for comparable events. This layer NEVER
infers physical Z/torque and NEVER compares inlet force to distal contact.
"""

from __future__ import annotations

import json
import math

import pytest

from lumen.experiments import phantom_metrics as pm


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _row(t, wire_ins=None, cath_ins=None, meas=None, sim=None, force=None):
    return {
        "t_s": t,
        "wire": [wire_ins, 0.0] if wire_ins is not None else None,
        "catheter": [cath_ins, 0.0] if cath_ins is not None else None,
        "measured_xy": {"wire": None, "catheter": None, **(meas or {})},
        "sim_xy": {"wire": None, "catheter": None, **(sim or {})},
        **({"inlet_force_n": force} if force is not None else {}),
    }


def _straight_rows(n=5, dt=1.0, speed=2.0):
    """Both instruments advance on the trunk, tips on the +Y axis."""
    rows = []
    for i in range(n):
        y = i * speed
        rows.append(
            _row(
                i * dt,
                wire_ins=y,
                cath_ins=y,
                meas={"catheter": [0.0, y], "wire": [0.0, y]},
                sim={"catheter": [0.0, y], "wire": [0.0, y]},
                force=0.1 if i >= 2 else None,
            )
        )
    return rows


def _branch_point(sign, along):
    """Point on the +/-35 deg daughter branch, `along` mm past the apex."""
    s = 1.0 if sign == "+" else -1.0
    a = math.radians(pm.BRANCH_ANGLE_DEG)
    return [s * along * math.sin(a), pm.BRANCH_APEX_Y_MM + along * math.cos(a)]


# ---------------------------------------------------------------------------
# Thresholds are predeclared in the module
# ---------------------------------------------------------------------------


def test_thresholds_predeclared():
    for name in (
        "XY_ERROR_MM_MAX",
        "BRANCH_ANGLE_DEG",
        "BRANCH_APEX_Y_MM",
        "BRANCH_RADIUS_MM",
        "BRANCH_SEGMENT_LENGTH_MM",
        "EVENT_TIME_S_TOL",
        "REVERSAL_DEADBAND_MM",
        "STRAIGHT_TRUNK_LATERAL_MM",
    ):
        assert hasattr(pm, name)
    assert pm.BRANCH_APEX_Y_MM == 50.0
    assert pm.BRANCH_ANGLE_DEG == 35.0
    assert pm.XY_ERROR_MM_MAX > 0 and pm.BRANCH_RADIUS_MM > 0


# ---------------------------------------------------------------------------
# Matched XY errors: ALL matched samples scored, per instrument
# ---------------------------------------------------------------------------


def test_perfect_agreement_zero_errors_no_discrepancy():
    out = pm.compare(_straight_rows())
    assert out["matched_xy_errors"]["catheter"]["count"] == 5
    assert out["matched_xy_errors"]["catheter"]["max_mm"] == pytest.approx(0.0)
    assert out["first_consequential_discrepancy"] is None
    json.dumps(out)  # JSON-serializable contract


def test_no_matches_yields_none_not_zero_error():
    rows = [_row(0.0, cath_ins=1.0, meas={"catheter": [0.0, 1.0]}, sim={})]
    out = pm.compare(rows)
    assert out["matched_xy_errors"]["catheter"]["count"] == 0
    assert out["matched_xy_errors"]["catheter"]["max_mm"] is None
    assert out["matched_xy_errors"]["catheter"]["mean_mm"] is None


def test_matched_only_null_side_excluded():
    rows = [_row(0.0, cath_ins=1.0, meas={"catheter": [0.0, 1.0]}, sim={})]
    out = pm.compare(rows)
    assert out["matched_xy_errors"]["catheter"]["count"] == 0
    assert out["coverage"]["matched_xy_rows"]["catheter"] == 0


def test_xy_errors_scored_for_all_matched_samples_even_in_branch():
    # One sample inside a daughter corridor must NOT be excluded: the branch
    # region is where the model is most likely wrong, so it is scored too.
    rows = [
        _row(0.0, cath_ins=30.0, meas={"catheter": [0.0, 30.0]}, sim={"catheter": [0.0, 30.0]}),
        _row(
            1.0,
            cath_ins=55.0,
            meas={"catheter": _branch_point("+", 5.0)},
            sim={"catheter": [_branch_point("+", 5.0)[0] + 6.0, _branch_point("+", 5.0)[1]]},
        ),
    ]
    out = pm.compare(rows)
    assert out["matched_xy_errors"]["catheter"]["count"] == 2
    assert out["matched_xy_errors"]["catheter"]["max_mm"] == pytest.approx(pm.XY_ERROR_MM_MAX + 1.0)


def test_per_instrument_tip_no_midstream_switching():
    # Catheter tip present, wire tip absent: wire events must be independent
    # and must NOT borrow the catheter tip.
    rows = []
    for i in range(6):
        t = float(i)
        y = min(10.0 * i, 50.0)
        m = {"catheter": _branch_point("+", 5.0)} if t >= 5.0 else {"catheter": [0.0, y]}
        rows.append(_row(t, wire_ins=y, cath_ins=y, meas=m, sim={"catheter": [0.0, y]}))
    out = pm.compare(rows)
    # Wire has no tip data at all -> no wire branch entry, no borrowing.
    assert out["branch_choice"]["wire"]["measured"] is None
    assert out["branch_choice"]["catheter"]["measured"]["sign"] == "+"
    assert "branch_choice" not in [
        d.get("event") for d in out["discrepancies"] if d.get("instrument") == "wire"
    ]


# ---------------------------------------------------------------------------
# Branch choice / entry time: corridor geometry, per instrument
# ---------------------------------------------------------------------------


def _branch_rows(meas_entry_t, sim_entry_t, sign="+", sim_sign=None, n=12):
    """Catheter advances to the apex, then (optionally) enters a daughter
    branch 4 mm past the apex (sub-XY-threshold offset). Wire stays on trunk."""
    sim_sign = sim_sign or sign
    rows = []
    for i in range(n):
        t = float(i)
        y = min(10.0 * i, 50.0)
        m = {"catheter": [0.0, y], "wire": [0.0, y]}
        sim = {"catheter": [0.0, y], "wire": [0.0, y]}
        if meas_entry_t is not None and t >= meas_entry_t:
            m["catheter"] = _branch_point(sign, 4.0)
        if sim_entry_t is not None and t >= sim_entry_t:
            sim["catheter"] = _branch_point(sim_sign, 4.0)
        rows.append(_row(t, wire_ins=y, cath_ins=y, meas=m, sim=sim))
    return rows


def test_matching_branch_entries_same_sign_no_discrepancy():
    out = pm.compare(_branch_rows(7.0, 7.0))
    bc = out["branch_choice"]["catheter"]
    assert bc["measured"]["t_s"] == pytest.approx(7.0)
    assert bc["measured"]["sign"] == "+"
    assert bc["measured"]["observed_at_s"] == pytest.approx(7.0)
    assert bc["sim"]["t_s"] == pytest.approx(7.0)
    assert bc["sim"]["sign"] == "+"
    assert bc["sim"]["observed_at_s"] == pytest.approx(7.0)
    assert out["branch_entry"]["catheter"]["signed_lag_s"] == pytest.approx(0.0)
    assert out["first_consequential_discrepancy"] is None


def test_branch_entry_per_instrument_and_both_tips_missing_reported():
    # Neither tip ever enters a branch: per-instrument records still exist
    # with a null lag (reported), and matched coverage uses matched counts.
    rows = _straight_rows()
    out = pm.compare(rows)
    for inst in pm.INSTRUMENTS:
        rec = out["branch_entry"][inst]
        assert rec["signed_lag_s"] is None
        assert rec["matched_rows"] == 5


def test_branch_sign_mismatch_is_first_discrepancy():
    out = pm.compare(_branch_rows(5.0, 5.0, sign="+", sim_sign="-"))
    d = out["first_consequential_discrepancy"]
    assert d is not None and d["event"] == "branch_choice"
    assert d["measured_sign"] == "+" and d["sim_sign"] == "-"
    assert d["t_s"] == pytest.approx(5.0)


def test_branch_sign_mismatch_reported_at_later_observation_despite_censoring():
    # Signs differ and one side's entry time is occlusion-censored: the
    # first-observed signs are still compared, at the later observation time.
    rows = _branch_rows(5.0, 6.0, sign="+", sim_sign="-")
    rows[0] = _row(0.0, cath_ins=0.0, meas={"catheter": [0.0, 0.0]}, sim={"catheter": [0.0, 0.0]})
    rows[1] = _row(1.0, cath_ins=30.0, meas={}, sim={})  # occluded before entry
    d = pm.compare(rows)["first_consequential_discrepancy"]
    assert d is not None and d["event"] == "branch_choice"
    assert d["measured_sign"] == "+" and d["sim_sign"] == "-"
    assert d["t_s"] == pytest.approx(6.0)


def test_branch_entry_delta_beyond_threshold_is_discrepancy():
    out = pm.compare(_branch_rows(5.0, 5.0 + pm.EVENT_TIME_S_TOL + 1.0))
    d = out["first_consequential_discrepancy"]
    assert d is not None and d["event"] == "branch_entry"
    assert d["signed_lag_s"] == pytest.approx(pm.EVENT_TIME_S_TOL + 1.0)


def test_pre_apex_lateral_point_is_not_a_branch_entry():
    # Laterally off-axis but BEFORE the apex: not in any daughter corridor.
    rows = []
    for i in range(6):
        rows.append(
            _row(
                float(i),
                cath_ins=10.0 * i,
                meas={"catheter": [3.0, 10.0 * i]},
                sim={"catheter": [3.0, 10.0 * i]},
            )
        )
    out = pm.compare(rows)
    assert out["branch_choice"]["catheter"]["measured"] is None
    assert out["first_consequential_discrepancy"] is None


def test_point_beyond_finite_segment_not_a_branch_entry():
    p = _branch_point("+", pm.BRANCH_SEGMENT_LENGTH_MM + 5.0)
    rows = [_row(0.0, cath_ins=60.0, meas={"catheter": p}, sim={"catheter": p})]
    out = pm.compare(rows)
    assert out["branch_choice"]["catheter"]["measured"] is None


def test_point_off_corridor_axis_not_a_branch_entry():
    # Past the apex but laterally far from both finite corridors.
    rows = [
        _row(
            0.0,
            cath_ins=60.0,
            meas={"catheter": [20.0, pm.BRANCH_APEX_Y_MM + 5.0]},
            sim={"catheter": [20.0, pm.BRANCH_APEX_Y_MM + 5.0]},
        )
    ]
    out = pm.compare(rows)
    assert out["branch_choice"]["catheter"]["measured"] is None


def test_first_sampled_branch_only():
    # Enter the '-' branch first, then move to '+': choice stays '-'.
    rows = [
        _row(0.0, cath_ins=40.0, meas={"catheter": [0.0, 40.0]}, sim={"catheter": [0.0, 40.0]}),
        _row(
            1.0,
            cath_ins=55.0,
            meas={"catheter": _branch_point("-", 5.0)},
            sim={"catheter": _branch_point("-", 5.0)},
        ),
        _row(
            2.0,
            cath_ins=56.0,
            meas={"catheter": _branch_point("+", 5.0)},
            sim={"catheter": _branch_point("+", 5.0)},
        ),
    ]
    out = pm.compare(rows)
    assert out["branch_choice"]["catheter"]["measured"]["sign"] == "-"
    assert out["first_consequential_discrepancy"] is None


def test_branch_entry_left_censored_reported_as_limit_not_discrepancy():
    out = pm.compare(_branch_rows(7.0, None))
    assert out["branch_choice"]["catheter"]["sim"] is None
    assert out["first_consequential_discrepancy"] is None
    assert "branch_entry" in out["coverage"]["excluded_events"]


def test_occlusion_before_first_event_nulls_timing_conservatively():
    # Instrument present but tip unobservable (occluded) before the entry:
    # timing must be null, never inferred across the gap.
    rows = [
        _row(0.0, cath_ins=0.0, meas={"catheter": [0.0, 0.0]}, sim={"catheter": [0.0, 0.0]}),
        _row(1.0, cath_ins=30.0, meas={}, sim={}),  # occluded
        _row(
            2.0,
            cath_ins=55.0,
            meas={"catheter": _branch_point("+", 5.0)},
            sim={"catheter": _branch_point("+", 5.0)},
        ),
    ]
    out = pm.compare(rows)
    bc = out["branch_choice"]["catheter"]
    assert bc["measured"]["sign"] == "+"
    assert bc["measured"]["t_s"] is None
    assert bc["measured"]["observed_at_s"] == pytest.approx(2.0)
    assert bc["sim"]["t_s"] is None
    assert bc["sim"]["observed_at_s"] == pytest.approx(2.0)
    assert out["branch_entry"]["catheter"]["signed_lag_s"] is None
    assert "branch_entry" in out["coverage"]["excluded_events"]
    assert out["first_consequential_discrepancy"] is None


# ---------------------------------------------------------------------------
# Reversal lag: per instrument, positive->negative insertion + straight-trunk
# tip reversal, deadband for jitter, signed lag only when comparable
# ---------------------------------------------------------------------------


def _reversal_rows(input_rev_i, meas_rev_i, sim_rev_i, n=12, dt=1.0, speed=2.0):
    """Insertion advances then reverses (positive -> negative) at rev indices."""

    def ins(i):
        if input_rev_i is None or i < input_rev_i:
            return speed * i
        return speed * input_rev_i - speed * (i - input_rev_i)

    def tip_y(i, rev_i):
        if rev_i is None or i < rev_i:
            return speed * i
        return speed * rev_i - speed * (i - rev_i)

    rows = []
    for i in range(n):
        rows.append(
            _row(
                i * dt,
                wire_ins=ins(i),
                cath_ins=ins(i),
                meas={"catheter": [0.0, tip_y(i, meas_rev_i)], "wire": [0.0, tip_y(i, meas_rev_i)]},
                sim={"catheter": [0.0, tip_y(i, sim_rev_i)], "wire": [0.0, tip_y(i, sim_rev_i)]},
            )
        )
    return rows


def test_reversal_agreement_no_discrepancy():
    out = pm.compare(_reversal_rows(5, 5, 5))
    rev = out["first_reversal"]["catheter"]
    assert rev["measured_t_s"] == pytest.approx(5.0)
    assert rev["sim_t_s"] == pytest.approx(5.0)
    assert rev["signed_lag_s"] == pytest.approx(0.0)
    assert out["first_consequential_discrepancy"] is None


def test_reversal_signed_lag_and_discrepancy():
    out = pm.compare(_reversal_rows(4, 4, 4 + pm.EVENT_TIME_S_TOL + 1))
    rev = out["first_reversal"]["catheter"]
    assert rev["signed_lag_s"] == pytest.approx(pm.EVENT_TIME_S_TOL + 1.0)
    events = [d for d in out["discrepancies"] if d["event"] == "first_reversal"]
    assert events and events[0]["signed_lag_s"] == pytest.approx(pm.EVENT_TIME_S_TOL + 1.0)


def test_reversal_earlier_in_sim_gives_negative_lag():
    out = pm.compare(_reversal_rows(6, 6, 4, n=14, speed=4.0))
    rev = out["first_reversal"]["catheter"]
    assert rev["signed_lag_s"] == pytest.approx(-2.0)


def test_decrease_without_prior_advance_is_not_a_reversal():
    # Insertion starts already positive and flat, then decreases: no prior
    # advance -> no reversal event on either side.
    rows = []
    for i, v in enumerate([5.0, 5.0, 5.0, 1.0, -1.0]):
        rows.append(
            _row(float(i), cath_ins=v, meas={"catheter": [0.0, v]}, sim={"catheter": [0.0, v]})
        )
    out = pm.compare(rows)
    assert out["first_reversal"]["catheter"]["measured_t_s"] is None
    assert out["first_consequential_discrepancy"] is None


def test_jitter_within_deadband_is_not_a_reversal():
    # Tip wobbles by less than REVERSAL_DEADBAND_MM: direction never flips.
    rows = []
    ys = [0.0, 2.0, 2.0 + 0.4 * pm.REVERSAL_DEADBAND_MM, 2.0, 4.0, 6.0]
    for i, y in enumerate(ys):
        ins = 2.0 + y
        rows.append(
            _row(float(i), cath_ins=ins, meas={"catheter": [0.0, y]}, sim={"catheter": [0.0, y]})
        )
    out = pm.compare(rows)
    assert out["first_reversal"]["catheter"]["measured_t_s"] is None


def test_reversal_never_observed_left_censored():
    out = pm.compare(_reversal_rows(None, 4, 4))
    assert out["first_reversal"]["catheter"]["measured_t_s"] is None
    assert out["first_consequential_discrepancy"] is None
    assert "first_reversal" in out["coverage"]["excluded_events"]


def test_reversal_on_curved_start_excluded():
    # Tip moves laterally before any longitudinal motion: not a straight trunk.
    rows = [
        _row(0.0, cath_ins=0.0, meas={"catheter": [0.0, 0.0]}, sim={"catheter": [0.0, 0.0]}),
        _row(1.0, cath_ins=0.5, meas={"catheter": [8.0, 0.0]}, sim={"catheter": [8.0, 0.0]}),
        _row(2.0, cath_ins=1.0, meas={"catheter": [16.0, 2.0]}, sim={"catheter": [16.0, 2.0]}),
        _row(3.0, cath_ins=0.5, meas={"catheter": [16.0, 1.0]}, sim={"catheter": [16.0, 1.0]}),
    ]
    out = pm.compare(rows)
    assert "first_reversal" in out["coverage"]["excluded_events"]
    assert out["first_consequential_discrepancy"] is None


def test_reversal_instruments_scored_independently():
    # Wire reverses at i=4, catheter never does: catheter excluded, wire scored.
    rows = []
    for i in range(10):
        w = 2.0 * i if i < 4 else 8.0 - 2.0 * (i - 4)
        c = 2.0 * i
        rows.append(
            _row(
                float(i),
                wire_ins=w,
                cath_ins=c,
                meas={"wire": [0.0, w], "catheter": [0.0, c]},
                sim={"wire": [0.0, w], "catheter": [0.0, c]},
            )
        )
    out = pm.compare(rows)
    assert out["first_reversal"]["wire"]["measured_t_s"] == pytest.approx(4.0)
    assert out["first_reversal"]["catheter"]["measured_t_s"] is None
    assert "first_reversal" in out["coverage"]["excluded_events"]


# ---------------------------------------------------------------------------
# First discrepancy: earliest across ALL instruments and events
# ---------------------------------------------------------------------------


def test_first_discrepancy_is_chronologically_first_event():
    # Wire XY error at t=1 precedes the catheter branch-entry lag at t>=6.
    rows = _branch_rows(7.0, 7.0 + pm.EVENT_TIME_S_TOL + 1.0)
    rows[1] = _row(
        1.0,
        wire_ins=10.0,
        cath_ins=10.0,
        meas={"wire": [0.0, 10.0], "catheter": [0.0, 10.0]},
        sim={"wire": [pm.XY_ERROR_MM_MAX + 2.0, 10.0], "catheter": [0.0, 10.0]},
    )
    d = pm.compare(rows)["first_consequential_discrepancy"]
    assert d["event"] == "matched_xy_error"
    assert d["instrument"] == "wire"
    assert d["t_s"] == pytest.approx(1.0)


def test_first_discrepancy_prefers_earlier_instrument_sample():
    # Catheter XY error at t=2, wire XY error at t=4: catheter wins on time,
    # not on instrument order.
    rows = _straight_rows(n=6)
    rows[2] = _row(
        2.0,
        wire_ins=4.0,
        cath_ins=4.0,
        meas={"catheter": [0.0, 4.0], "wire": [0.0, 4.0]},
        sim={"catheter": [pm.XY_ERROR_MM_MAX + 1.0, 4.0], "wire": [0.0, 4.0]},
    )
    rows[4] = _row(
        4.0,
        wire_ins=8.0,
        cath_ins=8.0,
        meas={"catheter": [0.0, 8.0], "wire": [0.0, 8.0]},
        sim={"catheter": [0.0, 8.0], "wire": [pm.XY_ERROR_MM_MAX + 1.0, 8.0]},
    )
    d = pm.compare(rows)["first_consequential_discrepancy"]
    assert d["instrument"] == "catheter"
    assert d["t_s"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Inlet force: availability only, never compared
# ---------------------------------------------------------------------------


def test_inlet_force_availability_only():
    rows = _straight_rows()
    out = pm.compare(rows)
    assert out["inlet_force_n"] == {"available_rows": 3, "total_rows": 5}
    assert all(d.get("event") != "inlet_force" for d in out["discrepancies"])


# ---------------------------------------------------------------------------
# General contract
# ---------------------------------------------------------------------------


def test_empty_rows_yields_clean_empty_report():
    out = pm.compare([])
    assert out["first_consequential_discrepancy"] is None
    for inst in pm.INSTRUMENTS:
        assert out["matched_xy_errors"][inst]["count"] == 0
    json.dumps(out)


def test_unsorted_rows_sorted_internally():
    rows = list(reversed(_straight_rows()))
    out = pm.compare(rows)
    assert out["matched_xy_errors"]["catheter"]["count"] == 5
    assert out["first_consequential_discrepancy"] is None


def test_rotation_deg_never_used_for_any_inference():
    a = _straight_rows()
    for i, r in enumerate(a):
        if r["catheter"]:
            r["catheter"][1] = 137.0 * (i + 1)
        if r["wire"]:
            r["wire"][1] = 91.0 * (i + 1)
    b = _straight_rows()
    assert pm.compare(a) == pm.compare(b)


def test_instruments_present_but_tips_absent_no_events():
    rows = [_row(float(i), wire_ins=2.0 * i, cath_ins=2.0 * i) for i in range(4)]
    out = pm.compare(rows)
    assert out["first_consequential_discrepancy"] is None
    assert out["coverage"]["matched_xy_rows"]["catheter"] == 0
