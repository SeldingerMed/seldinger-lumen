"""Phantom-layer comparison metrics over parent-instrumented rows (read-only).

This layer consumes rows produced by the parent instrumentation layer and
computes side-by-side measured-vs-sim comparison metrics. It NEVER touches
other layers and performs no I/O.

Row contract (per row)::

    {
        "t_s": float,
        "wire":     [signed_insertion_mm, rotation_deg] | None,
        "catheter": [signed_insertion_mm, rotation_deg] | None,
        "measured_xy": {"wire": [x, y] | None, "catheter": [x, y] | None},
        "sim_xy":      {"wire": [x, y] | None, "catheter": [x, y] | None},
        "inlet_force_n": float | None,          # optional
    }

Geometry: XY is the calibrated fork plane (Lumen X, Z). The procedural
bifurcation is fixed with its apex at (0, Y = 50 mm); the two daughter
branches leave the apex at +/-35 deg from the trunk (+Y) axis, and only a
FINITE segment of each is tracked (length BRANCH_SEGMENT_LENGTH_MM, corridor
radius BRANCH_RADIUS_MM). A tip is in a branch only when it lies inside one
corridor PAST the apex; pre-apex lateral wander is never a branch entry, and
once a tip has entered a corridor only that FIRST sampled branch is tracked.

Per-instrument independence: every instrument is scored on its OWN insertion
and its OWN tip XY. Tips are never substituted between instruments, and an
instrument with no tip data has no branch/reversal events at all.

Branch choice/entry: the first OBSERVED branch sign per side is always
comparable, reported with ``observed_at_s`` (first visible sample in the
branch) even when the true entry time is left-censored by a preceding
occlusion; sign mismatches are timestamped at the LATER observation time and
never claim a true first entry across gaps. ``branch_entry[inst]`` carries a
nullable ``signed_lag_s`` (unavailable when either entry time is censored or
a side never entered) plus a ``matched_rows`` coverage count of rows where
both sides' tips were observed; both tips missing still yields a reported
null-lag record. Conservative timing (no gap inference): a sample where the
instrument is present but its tip is unobservable is an occlusion. Any
occlusion strictly before a side's first event nulls that side's event
timing rather than inferring across the gap. Signed lags (sim_t - measured_t)
and residuals are reported only for comparable events (both sides observed,
neither nulled).

Reversals: per instrument, an insertion reversal is a positive->negative
crossing that required prior positive advance (a plain decrease from an
already-positive plateau is NOT a reversal), plus a longitudinal tip reversal
of the instrument's own tip Y on an initially straight trunk, with a jitter
deadband (REVERSAL_DEADBAND_MM) so sub-deadband wobble never flips direction.

Matched XY errors are scored for ALL matched samples, including samples inside
a daughter corridor — the branch region is not excluded from scoring.

Output is a JSON-serializable dict whose ``first_consequential_discrepancy``
is the chronologically earliest comparable discrepancy across ALL instruments
and event types (time order, not instrument order), or None with
coverage/limits attached.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- Predeclared thresholds -------------------------------------------------
XY_ERROR_MM_MAX = 5.0
EVENT_TIME_S_TOL = 2.0
BRANCH_APEX_Y_MM = 50.0
BRANCH_ANGLE_DEG = 35.0
BRANCH_RADIUS_MM = 1.6
BRANCH_SEGMENT_LENGTH_MM = 50.0
REVERSAL_DEADBAND_MM = 0.5
STRAIGHT_TRUNK_LATERAL_MM = 2.0

INSTRUMENTS = ("wire", "catheter")

# Daughter direction unit vectors indexed by branch sign.
_a = math.radians(BRANCH_ANGLE_DEG)
_DAUGHTER_DIR = {"+": (math.sin(_a), math.cos(_a)), "-": (-math.sin(_a), math.cos(_a))}


# --- Geometry ---------------------------------------------------------------


def _branch_sign(x: float, y: float) -> Optional[str]:
    """'+'/'-' if (x, y) lies inside a finite daughter corridor past the apex."""
    vx, vy = x, y - BRANCH_APEX_Y_MM
    if vy <= 0.0:
        return None
    matches = []
    for sign, (dx, dy) in _DAUGHTER_DIR.items():
        along = vx * dx + vy * dy
        if along <= 0.0 or along > BRANCH_SEGMENT_LENGTH_MM:
            continue
        perp = math.hypot(vx - along * dx, vy - along * dy)
        if perp <= BRANCH_RADIUS_MM:
            matches.append(sign)
    return matches[0] if len(matches) == 1 else None


# --- Per-instrument series --------------------------------------------------


def _series(
    rows: Sequence[Dict[str, Any]], inst: str, xy_key: Optional[str]
) -> List[Tuple[float, Optional[float], Optional[Tuple[float, float]]]]:
    """(t, insertion, tip) per row for ONE instrument; tip None = unobserved."""
    out = []
    for r in rows:
        v = r.get(inst)
        ins = float(v[0]) if v is not None else None
        tip = None
        if xy_key is not None:
            p = (r.get(xy_key) or {}).get(inst)
            if p is not None:
                tip = (float(p[0]), float(p[1]))
        out.append((float(r["t_s"]), ins, tip))
    return out


def _branch_entry(series) -> Optional[Tuple[str, Optional[float], float]]:
    """(sign, entry_t, observed_at_t) of the first sample inside a daughter
    corridor, or None. entry_t is None (conservatively) if the tip was
    occluded before that sample — the true first entry is left-censored —
    while observed_at_t is always the first VISIBLE sample in the branch.
    Only the FIRST sampled branch is tracked."""
    first: Optional[Tuple[str, Optional[float], float]] = None
    occluded = False
    for t, ins, tip in series:
        if tip is None:
            if ins is not None:
                occluded = True
            continue
        if first is None:
            s = _branch_sign(*tip)
            if s is not None:
                first = (s, None if occluded else t, t)
    return first


def _turn(series, *, tip_signal=False):
    """Last peak before confirmed >0.5 mm pullback after a >0.5 mm advance."""
    low = peak = peak_time = None
    advanced = False
    for t, insertion, tip in series:
        if tip_signal and (
            tip is None
            or abs(tip[0]) > STRAIGHT_TRUNK_LATERAL_MM
            or not 0 <= tip[1] < BRANCH_APEX_Y_MM
        ):
            return None
        value = tip[1] if tip_signal else insertion
        if value is None:
            return None
        if low is None or (not advanced and value < low):
            low = peak = value
            peak_time = t
        if value >= peak:
            peak, peak_time = value, t
        advanced |= peak - low > REVERSAL_DEADBAND_MM
        if advanced and peak - value > REVERSAL_DEADBAND_MM:
            return peak_time
    return None


# --- Public API -------------------------------------------------------------


def compare(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute phantom comparison metrics over time-sorted parent rows."""
    rows = sorted(rows, key=lambda r: r["t_s"])
    candidates: List[Tuple[float, int, Dict[str, Any]]] = []
    excluded: List[str] = []

    # -- matched XY errors: ALL matched samples, per instrument ---------------
    errors: Dict[str, Dict[str, Any]] = {}
    for inst in INSTRUMENTS:
        vals: List[float] = []
        for row in rows:
            m = (row.get("measured_xy") or {}).get(inst)
            s = (row.get("sim_xy") or {}).get(inst)
            if m is None or s is None:
                continue
            e = math.hypot(float(s[0]) - float(m[0]), float(s[1]) - float(m[1]))
            vals.append(e)
            if e > XY_ERROR_MM_MAX:
                candidates.append(
                    (
                        row["t_s"],
                        0,
                        {
                            "event": "matched_xy_error",
                            "instrument": inst,
                            "t_s": row["t_s"],
                            "error_mm": e,
                            "threshold_mm": XY_ERROR_MM_MAX,
                        },
                    )
                )
        errors[inst] = {
            "count": len(vals),
            "max_mm": max(vals) if vals else None,
            "mean_mm": (sum(vals) / len(vals)) if vals else None,
        }

    # -- branch choice / entry, per instrument --------------------------------
    # First OBSERVED signs are always comparable (reported at the later
    # observation time); entry timing is only comparable when neither side's
    # entry is left-censored by an occlusion before its first branch sample.
    branch_choice: Dict[str, Dict[str, Any]] = {}
    branch_entry: Dict[str, Dict[str, Any]] = {}
    for inst in INSTRUMENTS:
        meas = _branch_entry(_series(rows, inst, "measured_xy"))
        sim = _branch_entry(_series(rows, inst, "sim_xy"))
        branch_choice[inst] = {
            "measured": None
            if meas is None
            else {"sign": meas[0], "t_s": meas[1], "observed_at_s": meas[2]},
            "sim": None
            if sim is None
            else {"sign": sim[0], "t_s": sim[1], "observed_at_s": sim[2]},
        }
        # Coverage count uses the MATCHED count: rows where both sides'
        # tips were observed for this instrument.
        matched_rows = sum(
            1
            for r in rows
            if (r.get("measured_xy") or {}).get(inst) is not None
            and (r.get("sim_xy") or {}).get(inst) is not None
        )
        branch_entry[inst] = {"signed_lag_s": None, "matched_rows": matched_rows}
        if meas is None and sim is None:
            # Both tips missing: still reported above with a null lag.
            continue
        if meas is None or sim is None:
            excluded.append("branch_entry")
            continue
        if meas[0] != sim[0]:
            # Never claim a true first entry across gaps: timestamp the
            # sign mismatch at the later OBSERVATION time (always known).
            t0 = max(meas[2], sim[2])
            candidates.append(
                (
                    t0,
                    0,
                    {
                        "event": "branch_choice",
                        "instrument": inst,
                        "t_s": t0,
                        "measured_sign": meas[0],
                        "sim_sign": sim[0],
                    },
                )
            )
        else:
            mt, st = meas[1], sim[1]
            if mt is None or st is None:
                excluded.append("branch_entry")  # occlusion-nulled timing
            else:
                lag = st - mt
                branch_entry[inst]["signed_lag_s"] = lag
                if abs(lag) > EVENT_TIME_S_TOL:
                    candidates.append(
                        (
                            max(mt, st),
                            0,
                            {
                                "event": "branch_entry",
                                "instrument": inst,
                                "t_s": max(mt, st),
                                "signed_lag_s": lag,
                                "threshold_s": EVENT_TIME_S_TOL,
                            },
                        )
                    )

    # -- first reversal, per instrument ---------------------------------------
    first_reversal: Dict[str, Dict[str, Any]] = {}
    for inst in INSTRUMENTS:
        # Gate: a positive->negative measured-insertion reversal with prior
        # advance must exist, otherwise the reversal is left-censored.
        onset = _turn(_series(rows, inst, None))
        gate = onset is not None
        mt = _turn(_series(rows, inst, "measured_xy"), tip_signal=True) if gate else None
        st = _turn(_series(rows, inst, "sim_xy"), tip_signal=True) if gate else None
        rec = {
            "input_t_s": onset,
            "measured_t_s": mt,
            "sim_t_s": st,
            "measured_lag_s": None if mt is None else mt - onset,
            "sim_lag_s": None if st is None else st - onset,
            "insertion_reversed": gate,
            "signed_lag_s": None,
            "residual_s": None,
        }
        if not gate:
            excluded.append("first_reversal")  # left-censored insertion
            first_reversal[inst] = rec
            continue
        if mt is None and st is None:
            first_reversal[inst] = rec
            continue
        if mt is None or st is None:
            excluded.append("first_reversal")
            first_reversal[inst] = rec
            continue
        lag = st - mt
        rec["signed_lag_s"] = lag
        rec["residual_s"] = abs(lag)
        if abs(lag) > EVENT_TIME_S_TOL:
            candidates.append(
                (
                    max(mt, st),
                    0,
                    {
                        "event": "first_reversal",
                        "instrument": inst,
                        "t_s": max(mt, st),
                        "signed_lag_s": lag,
                        "threshold_s": EVENT_TIME_S_TOL,
                    },
                )
            )
        first_reversal[inst] = rec

    # -- chronologically first consequential discrepancy ----------------------
    candidates.sort(key=lambda c: (c[0], c[1]))
    discrepancies = [d for _, _, d in candidates]
    first = discrepancies[0] if discrepancies else None

    force_rows = [r for r in rows if r.get("inlet_force_n") is not None]
    return {
        "matched_xy_errors": errors,
        "branch_choice": branch_choice,
        "branch_entry": branch_entry,
        "first_reversal": first_reversal,
        "first_consequential_discrepancy": first,
        "discrepancies": discrepancies,
        "inlet_force_n": {
            "available_rows": len(force_rows),
            "total_rows": len(rows),
        },
        "coverage": {
            "matched_xy_rows": {
                inst: sum(
                    1
                    for r in rows
                    if (r.get("measured_xy") or {}).get(inst) is not None
                    and (r.get("sim_xy") or {}).get(inst) is not None
                )
                for inst in INSTRUMENTS
            },
            "excluded_events": excluded,
            "total_rows": len(rows),
            "thresholds": {
                "xy_error_mm_max": XY_ERROR_MM_MAX,
                "event_time_s_tol": EVENT_TIME_S_TOL,
                "branch_apex_y_mm": BRANCH_APEX_Y_MM,
                "branch_angle_deg": BRANCH_ANGLE_DEG,
                "branch_radius_mm": BRANCH_RADIUS_MM,
                "branch_segment_length_mm": BRANCH_SEGMENT_LENGTH_MM,
                "reversal_deadband_mm": REVERSAL_DEADBAND_MM,
                "straight_trunk_lateral_mm": STRAIGHT_TRUNK_LATERAL_MM,
            },
        },
    }
