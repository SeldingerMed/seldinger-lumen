"""Tests for the standalone lumen-rigid-pose/1 evaluator (score.py)."""

from __future__ import annotations

import json
import tempfile
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORE = REPO_ROOT / "lumen" / "registration_pack" / "score.py"
CASE = REPO_ROOT / "lumen" / "registration_pack" / "full-tree-v1"


def run_score(case_dir=CASE, estimate=None):
    if isinstance(estimate, (str, Path)):
        est_path = Path(estimate)
    else:
        with tempfile.TemporaryDirectory() as d:
            est_path = Path(d) / "estimate.json"
            est_path.write_text(json.dumps(estimate))
            return run_score(case_dir, est_path)
    cmd = [sys.executable, "-I", "-S", str(SCORE), str(case_dir), str(est_path)]
    return subprocess.run(cmd, capture_output=True, text=True)


def reference_pose():
    data = json.loads((CASE / "answer" / "pose.json").read_text())
    return np.array(data["model_to_camera_mm"], dtype=float)


def perturbed_pose(dx=3.0, dy=4.0, dz=0.0):
    """Reference pose composed with a transverse camera-frame translation."""
    delta = np.eye(4)
    delta[:3, 3] = [dx, dy, dz]
    return (delta @ reference_pose()).tolist()


def estimate(pose, case_id="full-tree-v1", schema="lumen-rigid-pose/1"):
    return {"schema": schema, "case_id": case_id, "model_to_camera_mm": pose}


def load_camera():
    return json.loads((CASE / "problem" / "problem.json").read_text())["camera"]


def load_points():
    geo = json.loads((CASE / "problem" / "geometry.json").read_text())
    return [p for b in geo["branches"] for p in b["points_mm"]]


def reproject(cam, T, points):
    T = np.array(T)
    pts = T[:3, :3] @ np.array(points).T + T[:3, 3:4]
    return (
        cam["cx_px"] + cam["fx_px"] * pts[0] / pts[2],
        cam["cy_px"] + cam["fy_px"] * pts[1] / pts[2],
    )


def expected_metrics(pose):
    cam, points = load_camera(), load_points()
    T_est = np.array(pose)
    T_ref = reference_pose()
    world = np.array(points).T
    a = T_ref[:3, :3] @ world + T_ref[:3, 3:4]
    b = T_est[:3, :3] @ world + T_est[:3, 3:4]
    rms3 = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=0))))
    ua, va = reproject(cam, T_ref.tolist(), points)
    ub, vb = reproject(cam, T_est.tolist(), points)
    rms2 = float(np.sqrt(np.mean((ua - ub) ** 2 + (va - vb) ** 2)))
    return rms3, rms2


def test_tracer_exact_reference():
    """Reference estimate scores ~0 and matches an independent computation."""
    pose = reference_pose().tolist()
    r = run_score(estimate=estimate(pose))
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    rms3, rms2 = expected_metrics(pose)
    assert out["case_id"] == "full-tree-v1"
    assert out["point_count"] == len(load_points())
    assert out["rms_3d_mm"] == pytest.approx(rms3, abs=1e-9)
    assert out["rms_reprojection_px"] == pytest.approx(rms2, abs=1e-6)
    assert out["rms_3d_mm"] < 1e-6


def test_transverse_perturbation_3_4_gives_5mm():
    """+3/+4 mm transverse translation yields exactly 5 mm 3D RMS error."""
    pose = perturbed_pose(3.0, 4.0)
    r = run_score(estimate=estimate(pose))
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["rms_3d_mm"] == pytest.approx(5.0, abs=1e-9)
    rms3, rms2 = expected_metrics(pose)
    assert out["rms_3d_mm"] == pytest.approx(rms3, abs=1e-9)
    assert out["rms_reprojection_px"] == pytest.approx(rms2, abs=1e-6)
    assert out["rms_reprojection_px"] > 0


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda e: e.update(schema="lumen-rigid-pose/2"), "schema"),
        (lambda e: e.update(case_id="other-case"), "case_id"),
        (lambda e: e.pop("model_to_camera_mm"), "keys"),
        (lambda e: e.update(extra=1), "keys"),
        (
            lambda e: e.update(
                model_to_camera_mm=[[True, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ),
            "numbers",
        ),
        (
            lambda e: e.update(
                model_to_camera_mm=[["0", 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ),
            "numbers",
        ),
        (
            lambda e: e.update(
                model_to_camera_mm=[
                    [float("nan"), 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
            "finite",
        ),
        (
            lambda e: e.update(
                model_to_camera_mm=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]]
            ),
            "last row",
        ),
        (
            lambda e: e.update(
                model_to_camera_mm=[[0.5, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ),
            "orthonormal",
        ),
        (
            lambda e: e.update(
                model_to_camera_mm=[[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ),
            "proper",
        ),
        (
            lambda e: e.update(
                model_to_camera_mm=[[1, 0.5, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ),
            "orthonormal",
        ),
        (lambda e: e.update(model_to_camera_mm=perturbed_pose(0, 0, -350.0)), "depth"),
    ],
)
def test_invalid_estimates_rejected(mutate, msg):
    est = estimate(perturbed_pose())
    mutate(est)
    r = run_score(estimate=est)
    assert r.returncode == 1, f"expected rejection ({msg}): {r.stdout}"
    assert msg in r.stderr


def test_depth_violation_rejected_by_pose_pushing_points_behind_source():
    pose = reference_pose().copy()
    pose[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ pose[:3, :3]
    pose[2, 3] = -5.0
    r = run_score(estimate=estimate(pose.tolist()))
    assert r.returncode == 1
    assert "depth" in r.stderr or "z" in r.stderr


def test_missing_estimate_file():
    r = run_score(estimate=CASE / "_does_not_exist.json")
    assert r.returncode == 1
    assert r.stderr.strip()


def test_usage_error():
    r = subprocess.run([sys.executable, "-I", "-S", str(SCORE)], capture_output=True, text=True)
    assert r.returncode == 2
    assert "usage" in r.stderr


def test_extreme_finite_translation_does_not_produce_nonfinite_score():
    pose = perturbed_pose()
    pose[0][3] = 1e308
    r = run_score(estimate=estimate(pose))
    assert r.returncode == 1 and "finite score" in r.stderr and not r.stdout
