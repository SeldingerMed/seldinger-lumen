"""Strict normalized capture-export admission for phantom capture runs.

`admit(run)` validates a capture export and returns normalized rows, one per
timestamp. Everything that can be wrong is rejected with :class:`AdmissionError`
— admission is a gate, not a repair pass.

Run shape (dict):
  metadata:
    run_id, recording_id, phantom_id : non-empty str
    provenance     : 'synthetic' | 'authorized_phantom'
    consent        : bool (True)
    tool_ids       : dict of tool identifiers; must include a 'wire' tool id
    calibration_id : non-empty str
    homography     : non-singular 3x3 matrix (list of 3 rows x 3 numbers),
                     pixel -> Lumen fork-plane XY mm
    clock_sync     : dict with non-empty 'evidence' and 'reference'
    units          : must exactly match the canonical map
                     {'insertion': 'mm', 'rotation': 'deg_unwrapped',
                      't_s': 's', 'measured_xy': 'mm', 'inlet_force_n': 'N'}
    motion_exit    : dict — independently measured motion at tool exit
                     (source 'external_tracker' or 'tool_exit' plus
                     non-empty 'evidence'; never motor commands). May be
                     omitted/None ONLY when no row carries tool motion;
                     rows with wire/catheter motion require it regardless.
    inlet_sensor   : required when any row carries a non-null
                     inlet_force_n — {'site': must name the explicit inlet
                     axial site, 'calibration_id': non-empty str}. Never a
                     torque-Z derivation.
    measured_xy_source : required (non-empty str) when any row carries a
                     calibrated XY without raw pixels — documents where the
                     XY came from
    units          : canonical units, keyed by field name
                     ('t_s': 's', 'measured_xy': 'mm', 'inlet_force_n': 'N')
    signs          : {'wire': [±1, ±1], 'catheter': [±1, ±1]} — sign
                     convention applied downstream by the solver; admit only
                     validates and preserves raw values (negative raw
                     insertion is valid when the polarity sign is -1).
  rows: 1..3001 dicts, each:
    t_s           : finite float, strictly increasing, >= 0  (clock monotonic)
    wire          : [abs_mm, unwrapped_degrees] floats, or None
    catheter      : same, or None (null when not visible)
    command       : logged only — any JSON-able value or None (NOT evidence)
    measured_xy   : {'wire': [x, y] mm | None, 'catheter': [x, y] mm | None};
                    if a pair is present both coords must be finite floats
                    (partial measurements rejected)
    inlet_force_n : finite float or None — explicit inlet axial site, Newtons;
                    signed only when instrumented. Never implied from a
                    torque-Z reading.
    visibility    : raw per-tip visibility flags (preserved verbatim)
    confidence    : raw per-tip confidence, each None or in [0, 1]
    raw_pixel     : optional per-tip raw pixel coords (preserved verbatim)

Projective consistency: a non-null raw_pixel must map through the normalized
homography onto the paired measured_xy. The homography is normalized by its
signed largest-magnitude element (see :func:`normalized_homography`), so
admission is invariant to the matrix's overall ± scale; the normalized form
is exported for stable parent-side hashing.

A tracked device may be visible without motion instrumentation (e.g. a
stationary catheter observed in fluoroscopy): wire/catheter may carry
translation while inlet_force_n is null. Units are canonical and live in the
key names (mm, degrees, s, N).
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "AdmissionError",
    "MAX_RUN_LENGTH",
    "admit",
    "normalized_homography",
    "synthetic_run",
]

MAX_RUN_LENGTH = 3001
MAX_WIRE_MM = 49.0
MAX_ROLL_ABS_DEG = 360.0

_PROJECTIVE_TOL_MM = 1e-3
_MIN_HOMOGRAPHY_DET = 1e-9
_MAX_ABS_VALUE = 1e12  # huge magnitudes are unsupported numeric input

_CANONICAL_UNITS = {
    "insertion": "mm",
    "rotation": "deg_unwrapped",
    "t_s": "s",
    "measured_xy": "mm",
    "inlet_force_n": "N",
}

_PROVENANCES = ("synthetic", "authorized_phantom")
_MOTION_EXIT_SOURCES = ("external_tracker", "tool_exit")
# calibration placeholders rejected per BenchTraceMetadata semantics
_CALIBRATION_PLACEHOLDERS = {"none", "unknown", "uncalibrated"}


class AdmissionError(ValueError):
    """A capture export failed strict admission."""


def _is_num(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and abs(x) <= _MAX_ABS_VALUE
        and math.isfinite(x)
    )


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AdmissionError(msg)


def _nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _is_calibration_placeholder(x: Any) -> bool:
    return isinstance(x, str) and x.strip().lower() in _CALIBRATION_PLACEHOLDERS


def _check_calibration_id(value: Any, label: str) -> None:
    _require(_nonempty_str(value), f"{label} must be a non-empty string")
    _require(
        not _is_calibration_placeholder(value),
        f"{label} must identify a calibration (placeholders "
        f"{sorted(_CALIBRATION_PLACEHOLDERS)} rejected, per BenchTraceMetadata)",
    )


def normalized_homography(matrix: Any) -> list[list[float]] | None:
    """Scale-invariant normalized form of a 3x3 homography matrix.

    Divides every element by the signed element of largest magnitude, so
    ``matrix`` and any ± scale of it produce the identical result — suitable
    as a stable canonical form for hashing. Returns ``None`` when the input
    is not a finite 3x3 numeric matrix or is identically zero (admission
    treats ``None`` as degenerate and rejects it).
    """
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 3:
        return None
    rows: list[list[float]] = []
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return None
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in row
        ):
            return None
        rows.append([float(v) for v in row])
    flat = [v for row in rows for v in row]
    pivot = max(flat, key=abs)
    if pivot == 0.0:
        return None
    return [[v / pivot for v in row] for row in rows]


def _check_homography(hom: Any) -> None:
    norm = normalized_homography(hom)
    _require(
        norm is not None,
        "homography must be a 3x3 finite numeric matrix (pixel -> fork-plane XY mm)",
    )
    if norm is None:
        return
    (a, b, c) = norm
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    _require(
        _is_num(det) and abs(det) > _MIN_HOMOGRAPHY_DET,
        "homography must be non-singular (invertible pixel -> XY mm map)",
    )


def _check_metadata(meta: Any) -> None:
    _require(isinstance(meta, dict), "metadata must be a dict")
    for key in (
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
    ):
        _require(key in meta, f"metadata missing {key!r}")
    for key in ("run_id", "recording_id", "phantom_id"):
        _require(_nonempty_str(meta[key]), f"{key} must be a non-empty string")
    _check_calibration_id(meta["calibration_id"], "calibration_id")
    _require(
        meta["provenance"] in _PROVENANCES, "provenance must be 'synthetic' or 'authorized_phantom'"
    )
    _require(meta["consent"] is True, "consent must be True")
    tool_ids = meta["tool_ids"]
    _require(
        isinstance(tool_ids, dict) and _nonempty_str(tool_ids.get("wire")),
        "tool_ids must be a dict containing a non-empty 'wire' tool id",
    )
    _check_homography(meta["homography"])
    clock_sync = meta["clock_sync"]
    _require(
        isinstance(clock_sync, dict)
        and _nonempty_str(clock_sync.get("evidence"))
        and _nonempty_str(clock_sync.get("reference")),
        "clock_sync must include non-empty 'evidence' and 'reference'",
    )
    units = meta["units"]
    _require(
        units == _CANONICAL_UNITS,
        f"units must exactly match the canonical map {_CANONICAL_UNITS}",
    )
    signs = meta["signs"]
    _require(
        isinstance(signs, dict) and set(signs) == {"wire", "catheter"},
        "signs must have exactly 'wire' and 'catheter' keys",
    )
    for tool in ("wire", "catheter"):
        pair = signs[tool]
        _require(
            isinstance(pair, (list, tuple))
            and len(pair) == 2
            and all(isinstance(s, int) and not isinstance(s, bool) and s in (1, -1) for s in pair),
            f"signs[{tool!r}] must be [±1, ±1]",
        )
    motion_exit = meta.get("motion_exit")
    if motion_exit is not None:
        _require(
            isinstance(motion_exit, dict),
            "motion_exit must be a dict (independently measured motion at tool exit), or be "
            "omitted entirely when no row carries tool motion",
        )
        _require(
            motion_exit.get("source") in _MOTION_EXIT_SOURCES
            and _nonempty_str(motion_exit.get("evidence")),
            "motion_exit requires independent 'external_tracker' or 'tool_exit' source with non-empty "
            "'evidence' (motor commands are not evidence)",
        )


def _check_tool_state(state: Any, label: str) -> None:
    if state is None:
        return
    _require(
        isinstance(state, (list, tuple)) and len(state) == 2 and all(map(_is_num, state)),
        f"{label} must be [abs_mm, unwrapped_degrees] or null",
    )
    _require(abs(state[1]) <= MAX_ROLL_ABS_DEG, f"{label} unwrapped_degrees out of range")


def _check_xy_pair(pair: Any, label: str) -> None:
    if pair is None:
        return
    _require(
        isinstance(pair, (list, tuple)) and len(pair) == 2 and all(map(_is_num, pair)),
        f"measured_xy[{label!r}] must be a finite [x, y] mm pair or null (partial measurements rejected)",
    )


def _check_pixel_pair(pair: Any, label: str) -> None:
    if pair is None:
        return
    _require(
        isinstance(pair, (list, tuple)) and len(pair) == 2 and all(map(_is_num, pair)),
        f"raw_pixel[{label!r}] must be a finite [u, v] pixel pair or null",
    )


def _check_row(row: Any) -> None:
    _require(isinstance(row, dict), "each row must be a dict")
    for key in (
        "t_s",
        "wire",
        "catheter",
        "command",
        "measured_xy",
        "inlet_force_n",
        "visibility",
        "confidence",
    ):
        _require(key in row, f"row missing {key!r}")
    _require(_is_num(row["t_s"]) and row["t_s"] >= 0.0, "t_s must be a finite, non-negative float")
    _check_tool_state(row["wire"], "wire")
    _check_tool_state(row["catheter"], "catheter")
    _require(
        row["command"] is None or isinstance(row["command"], (str, int, float, bool, list, dict)),
        "command must be JSON-able or None (logged only)",
    )
    mxy = row["measured_xy"]
    _require(
        isinstance(mxy, dict) and set(mxy) == {"wire", "catheter"},
        "measured_xy must have exactly 'wire' and 'catheter' keys",
    )
    _check_xy_pair(mxy["wire"], "wire")
    _check_xy_pair(mxy["catheter"], "catheter")
    _require(
        row["inlet_force_n"] is None or _is_num(row["inlet_force_n"]),
        "inlet_force_n must be a finite float (N, explicit inlet axial site) or null",
    )
    visibility = row["visibility"]
    _require(
        isinstance(visibility, dict)
        and set(visibility) == {"wire", "catheter"}
        and all(isinstance(v, bool) for v in visibility.values()),
        "visibility must be a dict with boolean 'wire' and 'catheter' flags",
    )
    confidence = row["confidence"]
    _require(
        isinstance(confidence, dict) and set(confidence) == {"wire", "catheter"},
        "confidence must be a dict with 'wire' and 'catheter' keys (validated per tip)",
    )
    for tool, value in confidence.items():
        _require(
            value is None or (_is_num(value) and 0.0 <= value <= 1.0),
            f"confidence[{tool!r}] must be None or a number in [0, 1]",
        )
    raw_pixel = row.get("raw_pixel")
    if raw_pixel is not None:
        _require(
            isinstance(raw_pixel, dict) and set(raw_pixel) == {"wire", "catheter"},
            "raw_pixel must have exactly 'wire' and 'catheter' keys",
        )
        _check_pixel_pair(raw_pixel["wire"], "wire")
        _check_pixel_pair(raw_pixel["catheter"], "catheter")


def _project_through_homography(hom: Any, uv: list) -> tuple:
    u, v = float(uv[0]), float(uv[1])
    w = hom[2][0] * u + hom[2][1] * v + hom[2][2]
    _require(
        _is_num(w) and abs(w) > _MIN_HOMOGRAPHY_DET,
        "homography projects raw_pixel to a degenerate point",
    )
    x = (hom[0][0] * u + hom[0][1] * v + hom[0][2]) / w
    y = (hom[1][0] * u + hom[1][1] * v + hom[1][2]) / w
    return x, y


def _check_rows(rows: list, meta: dict) -> None:
    hom = normalized_homography(meta["homography"])
    signs = meta["signs"]
    needs_xy_source = False
    has_inlet = False
    has_catheter = False
    has_motion = False
    for i, row in enumerate(rows):
        try:
            _check_row(row)
        except AdmissionError as exc:
            raise AdmissionError(f"row {i}: {exc}") from None
        _require(
            i == 0 or row["t_s"] > rows[i - 1]["t_s"],
            f"row {i}: t_s must be strictly increasing (clock monotonic)",
        )
        raw_pixel = row.get("raw_pixel") or {}
        for tool in ("wire", "catheter"):
            visible = row["visibility"][tool]
            xy = row["measured_xy"][tool]
            pix = raw_pixel.get(tool)
            if not visible:
                _require(xy is None, f"row {i}: measured_xy[{tool!r}] must be null while invisible")
                _require(pix is None, f"row {i}: raw_pixel[{tool!r}] must be null while invisible")
            if pix is not None:
                _require(
                    xy is not None,
                    f"row {i}: raw_pixel[{tool!r}] present but measured_xy[{tool!r}] is null",
                )
                x, y = _project_through_homography(hom, pix)
                _require(
                    math.isclose(x, xy[0], abs_tol=_PROJECTIVE_TOL_MM)
                    and math.isclose(y, xy[1], abs_tol=_PROJECTIVE_TOL_MM),
                    f"row {i}: raw_pixel[{tool!r}] does not project onto measured_xy[{tool!r}] "
                    "through the homography",
                )
            elif xy is not None:
                needs_xy_source = True  # calibrated XY without raw pixels
            state = row[tool]
            if state is not None:
                has_motion = True
                _require(
                    0 <= state[0] * signs[tool][0] <= MAX_WIRE_MM,
                    f"row {i}: signed insertion outside replay bounds for {tool}",
                )
            if tool == "catheter" and (state is not None or xy is not None):
                has_catheter = True
        if row["inlet_force_n"] is not None:
            has_inlet = True
    if has_catheter:
        _require(
            _nonempty_str(meta["tool_ids"].get("catheter")),
            "rows with catheter motion or measured XY require a non-empty 'catheter' tool id",
        )
    if has_motion and meta.get("motion_exit") is None:
        raise AdmissionError(
            "tool motion present requires independent motion_exit evidence ('external_tracker' or "
            "'tool_exit' source with non-empty 'evidence') even when the field is omitted"
        )
    if needs_xy_source:
        _require(
            _nonempty_str(meta.get("measured_xy_source")),
            "calibrated measured_xy without raw pixels requires non-empty metadata "
            "'measured_xy_source' (documented XY provenance)",
        )
    if has_inlet:
        sensor = meta.get("inlet_sensor")
        _require(
            isinstance(sensor, dict) and sensor.get("site") == "inlet_axial",
            "non-null inlet_force_n requires metadata 'inlet_sensor' with the explicit inlet site "
            "and a non-empty 'calibration_id'",
        )
        if isinstance(sensor, dict):
            _check_calibration_id(sensor.get("calibration_id"), "inlet_sensor['calibration_id']")


def admit(run: Any) -> list[dict[str, Any]]:
    """Validate a capture export and return the normalized rows (same dicts)."""
    _require(isinstance(run, dict), "run must be a dict")
    _require("metadata" in run and "rows" in run, "run must have 'metadata' and 'rows'")
    _check_metadata(run["metadata"])
    rows = run["rows"]
    _require(isinstance(rows, list), "rows must be a list")
    _require(1 <= len(rows) <= MAX_RUN_LENGTH, f"run length must be 1..{MAX_RUN_LENGTH} rows")
    _check_rows(rows, run["metadata"])
    return rows


# ------------------------------------------------------------ example factory


def synthetic_run(
    run_id: str, recording_id: str, variant: str = "straight", n: int = 5
) -> dict[str, Any]:
    """Tiny self-admitting example factory for tests and fixtures.

    Rows are REPLAYABLE: zero-start, strictly 20 ms sampling, analytic
    measurements with homography-consistent raw pixels.

    variant='straight': wire advances linearly, catheter stationary and
    coaxial with the wire tip; inlet force instrumented.
    variant='partial': wire only (catheter untracked), no inlet force, one
    visible-but-missing frame, and a known heldout discrepancy (> 5 mm
    longitudinal) injected on the last sample.
    """
    if variant not in ("straight", "partial"):
        raise ValueError("variant must be 'straight' or 'partial'")
    n = max(1, n)
    missing_frame_idx = 2 if (variant == "partial" and n > 3) else None
    rows = []
    for i in range(n):
        t = round(0.02 * i, 6)
        if variant == "straight":
            adv, rot = 2.0 * i, 0.0
            wire_xy = [1.45 + 0.5 * i, 17.0]
            catheter = [0.0, 0.0]
            inlet = 0.05 * i
            visibility = {"wire": True, "catheter": True}
            confidence = {"wire": 0.95, "catheter": 0.8}
        else:
            adv, rot = 1.0 * i, 3.0 * i
            wire_xy = [1.45 + 0.1 * i, 17.0 + 0.05 * i]
            if i == n - 1:
                wire_xy = [wire_xy[0], wire_xy[1] + 18.0]  # heldout discrepancy
            catheter = None
            inlet = None
            visibility = {"wire": i != missing_frame_idx, "catheter": False}
            confidence = {"wire": 0.9, "catheter": None}
        missing = i == missing_frame_idx
        # missing frame: flag says visible, but measurement is absent
        if missing:
            visibility = {"wire": True, "catheter": False}
            confidence = {"wire": None, "catheter": None}
        rows.append(
            {
                "t_s": t,
                "wire": [float(adv), float(rot)],
                "catheter": catheter,
                "command": {"advance_mm": 2.0 if variant == "straight" else 1.0},
                "measured_xy": {
                    "wire": None if missing else [float(wire_xy[0]), float(wire_xy[1])],
                    "catheter": None
                    if catheter is None
                    else [float(wire_xy[0]), float(wire_xy[1])],
                },
                "inlet_force_n": inlet,
                "visibility": visibility,
                "confidence": confidence,
                "raw_pixel": {
                    "wire": None if missing else [float(wire_xy[0]), float(wire_xy[1])],
                    "catheter": None
                    if catheter is None
                    else [float(wire_xy[0]), float(wire_xy[1])],
                },
            }
        )
    metadata = {
        "run_id": run_id,
        "recording_id": recording_id,
        "phantom_id": "phantom-a",
        "provenance": "synthetic",
        "consent": True,
        "tool_ids": {"wire": "wire-0", "catheter": "cath-0"},
        "calibration_id": "cal-2026-11",
        "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "clock_sync": {"evidence": "ptp-offset-0.2ms", "reference": "capture-host"},
        "motion_exit": {"source": "tool_exit", "evidence": "synthetic independent encoder"},
        "signs": {"wire": [1, 1], "catheter": [1, 1]},
        "units": {
            "insertion": "mm",
            "rotation": "deg_unwrapped",
            "t_s": "s",
            "measured_xy": "mm",
            "inlet_force_n": "N",
        },
    }
    if variant == "straight":
        metadata["inlet_sensor"] = {"site": "inlet_axial", "calibration_id": "cal-2026-11"}
    return {"metadata": metadata, "rows": rows}
