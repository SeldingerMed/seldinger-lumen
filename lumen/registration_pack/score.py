"""Standalone evaluator for lumen-rigid-pose/1 estimates.

Usage: python -I -S score.py CASE_DIR ESTIMATE_JSON

Reads the trusted case (problem/problem.json, problem/geometry.json,
answer/pose.json) and an estimate JSON, validates the estimate against the
exact closed schema, and reports case_id, point_count, rms_3d_mm and
rms_reprojection_px. Standard library only; exits non-zero on any
validation failure (reason on stderr).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SCHEMA = "lumen-rigid-pose/1"
ORTHO_TOL = 3e-6  # Allow rotation entries rounded to six decimal places.


class ValidationError(Exception):
    pass


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _load_json(path: Path, what: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ValidationError(f"missing {what}: {path}")
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {what}: {exc}")
    if not isinstance(data, dict):
        raise ValidationError(f"{what} must be a JSON object")
    return data


def _validate_camera(problem: dict) -> None:
    cam = problem.get("camera")
    if not isinstance(cam, dict):
        raise ValidationError("problem.camera missing")
    axes = cam.get("axes", "")
    for token in ("+x right", "+y down", "+z forward"):
        if not isinstance(axes, str) or token not in axes:
            raise ValidationError(f"camera axes must state {token!r}: {axes!r}")
    for key in (
        "fx_px",
        "fy_px",
        "cx_px",
        "cy_px",
        "width_px",
        "height_px",
        "source_to_detector_mm",
    ):
        if not _is_number(cam.get(key)) or not math.isfinite(cam[key]):
            raise ValidationError(f"camera.{key} must be a finite number")
    if cam["fx_px"] <= 0 or cam["fy_px"] <= 0:
        raise ValidationError("camera focal lengths must be positive")
    if cam["source_to_detector_mm"] <= 0:
        raise ValidationError("camera.source_to_detector_mm must be positive")
    if not (
        _is_number(cam["width_px"])
        and _is_number(cam["height_px"])
        and cam["width_px"] > 0
        and cam["height_px"] > 0
    ):
        raise ValidationError("camera image dimensions must be positive")


def _validate_rigid(T, what: str):
    """Validate a 4x4 finite rigid transform; return (R, t)."""
    if not isinstance(T, list) or len(T) != 4:
        raise ValidationError(f"{what} must be a 4x4 matrix")
    for row in T:
        if not isinstance(row, list) or len(row) != 4:
            raise ValidationError(f"{what} must be a 4x4 matrix")
        for x in row:
            if not _is_number(x):
                raise ValidationError(f"{what} entries must be numbers (not bool/str)")
            if not math.isfinite(x):
                raise ValidationError(f"{what} entries must be finite")
    if T[3] != [0.0, 0.0, 0.0, 1.0]:
        raise ValidationError(f"{what} last row must be exactly [0, 0, 0, 1]")
    R = [[float(T[i][j]) for j in range(3)] for i in range(3)]
    # orthonormality: R^T R == I
    for i in range(3):
        for j in range(3):
            dot = sum(R[k][i] * R[k][j] for k in range(3))
            target = 1.0 if i == j else 0.0
            if abs(dot - target) > ORTHO_TOL:
                raise ValidationError(f"{what} rotation is not orthonormal (shear/scale)")
    det = (
        R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
        - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
        + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0])
    )
    if abs(det - 1.0) > ORTHO_TOL:
        raise ValidationError(f"{what} rotation must be proper (det=+1, no reflection)")
    t = [float(T[i][3]) for i in range(3)]
    return R, t


def _apply(R, t, p):
    return [R[i][0] * p[0] + R[i][1] * p[1] + R[i][2] * p[2] + t[i] for i in range(3)]


def score(case_dir: str, estimate_path: str) -> dict:
    root = Path(case_dir)
    problem = _load_json(root / "problem" / "problem.json", "problem.json")
    if problem.get("schema") != "lumen-registration/1":
        raise ValidationError("problem.json schema must be lumen-registration/1")
    _validate_camera(problem)

    geometry = _load_json(root / "problem" / "geometry.json", "geometry.json")
    branches = geometry.get("branches")
    if not isinstance(branches, list) or not branches:
        raise ValidationError("geometry.json must contain branches")
    points = []
    for branch in branches:
        pts = branch.get("points_mm") if isinstance(branch, dict) else None
        if not isinstance(pts, list):
            raise ValidationError("each branch must list points_mm")
        for p in pts:
            if (
                not isinstance(p, list)
                or len(p) != 3
                or not all(_is_number(c) for c in p)
                or not all(math.isfinite(c) for c in p)
            ):
                raise ValidationError("points_mm entries must be finite [x, y, z] triples")
            points.append([float(c) for c in p])
    if not points:
        raise ValidationError("geometry contains no points")

    answer = _load_json(root / "answer" / "pose.json", "answer/pose.json")
    if answer.get("schema") != SCHEMA:
        raise ValidationError("answer pose schema must be lumen-rigid-pose/1")
    case_id = problem.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValidationError("problem.case_id missing")
    if answer.get("case_id") != case_id:
        raise ValidationError("answer case_id does not match problem case_id")
    R_ref, t_ref = _validate_rigid(answer.get("model_to_camera_mm"), "answer model_to_camera_mm")

    est = _load_json(estimate_path, "estimate")
    if set(est.keys()) != {"schema", "case_id", "model_to_camera_mm"}:
        raise ValidationError("estimate keys must be exactly {schema, case_id, model_to_camera_mm}")
    if est["schema"] != SCHEMA:
        raise ValidationError(f"estimate schema must be {SCHEMA!r}")
    if est["case_id"] != case_id:
        raise ValidationError("estimate case_id does not match problem case_id")
    R_est, t_est = _validate_rigid(est["model_to_camera_mm"], "estimate model_to_camera_mm")

    fx = float(problem["camera"]["fx_px"])
    fy = float(problem["camera"]["fy_px"])
    cx = float(problem["camera"]["cx_px"])
    cy = float(problem["camera"]["cy_px"])
    sdd = float(problem["camera"]["source_to_detector_mm"])

    def project(point):
        X, Y, Z = point
        if not (0 < Z <= sdd):
            raise ValidationError("point depth outside 0 < z <= source_to_detector_mm")
        return cx + fx * X / Z, cy + fy * Y / Z

    n = len(points)
    sq3 = sq2 = 0.0
    for p in points:
        a, b = _apply(R_ref, t_ref, p), _apply(R_est, t_est, p)
        d3 = math.dist(a, b)
        d2 = math.dist(project(a), project(b))
        sq3 += d3 * d3
        sq2 += d2 * d2
    if not all(map(math.isfinite, (sq2, sq3))):
        raise ValidationError("pose must yield a finite score")

    return {
        "case_id": case_id,
        "point_count": n,
        "rms_3d_mm": math.sqrt(sq3 / n),
        "rms_reprojection_px": math.sqrt(sq2 / n),
    }


def main(argv) -> int:
    if len(argv) != 3:
        print("usage: python -I -S score.py CASE_DIR ESTIMATE_JSON", file=sys.stderr)
        return 2
    try:
        result = score(argv[1], argv[2])
    except (ValidationError, OSError, ValueError, OverflowError) as exc:
        print(f"invalid estimate: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
