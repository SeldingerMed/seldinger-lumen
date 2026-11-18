"""Tests for the procedural full-tree registration case generator (layer 1).

The generator is ``lumen.registration_pack.generate``; it renders a synthetic
branching vessel tree through a true ``lumen.sensors.carm.CArm`` (divergent beam,
source at origin, +x right / +y down / +z forward, screen y down after flipping
the native projected row) and emits a problem/ + answer/ case pack.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from lumen.registration_pack import generate as gen
from lumen.sensors.carm import CArm

REQUIRED_PROBLEM = {"image.png", "geometry.json", "problem.json"}
REQUIRED_ANSWER = {"pose.json", "overlay.png", "territories.json"}


def _generate(tmp_path: Path, **kw):
    out = tmp_path / "case"
    gen.generate_case(out, **kw)
    return out


# ---------------------------------------------------------------- tracer ----


def test_generated_files_exist(tmp_path):
    out = _generate(tmp_path)
    for name in REQUIRED_PROBLEM:
        assert (out / "problem" / name).is_file(), name
    for name in REQUIRED_ANSWER:
        assert (out / "answer" / name).is_file(), name
    masks = json.loads((out / "answer" / "territories.json").read_text())
    assert masks, "territory map must not be empty"
    for territory, fname in masks.items():
        assert (out / "answer" / fname).is_file(), (territory, fname)


# -------------------------------------------------------------- geometry ----


def test_geometry_tree_schema_and_connectivity(tmp_path):
    out = _generate(tmp_path)
    geom = json.loads((out / "problem" / "geometry.json").read_text())
    branches = geom["branches"]
    assert len(branches) >= 10  # multilevel branching tree

    ids = {b["id"] for b in branches}
    parents = {b["id"]: b["parent"] for b in branches}
    roots = [i for i, p in parents.items() if p is None]
    assert len(roots) == 1
    for b in branches:
        assert b["parent"] is None or b["parent"] in ids
        assert isinstance(b["territory"], str) and b["territory"]
        pts = np.asarray(b["points_mm"], float)
        assert pts.shape[1] == 3 and len(pts) >= 2
        assert np.isfinite(pts).all()
        assert not np.allclose(pts[0], pts[-1])  # curved, not degenerate
        assert isinstance(b["radius_mm"], (int, float)) and b["radius_mm"] > 0
    # every non-root reachable from the single root (connected, acyclic)
    for b in branches:
        seen, cur = set(), b["id"]
        while parents[cur] is not None:
            assert cur not in seen
            seen.add(cur)
            cur = parents[cur]
        assert cur == roots[0]


def test_geometry_nonplanar_and_asymmetric(tmp_path):
    out = _generate(tmp_path)
    geom = json.loads((out / "problem" / "geometry.json").read_text())
    allpts = np.concatenate([np.asarray(b["points_mm"], float) for b in geom["branches"]])
    # nonplanar: points must not fit any single plane well
    c = allpts - allpts.mean(0)
    sv = np.linalg.svd(c, compute_uv=False)
    assert sv[2] / sv[0] > 0.05, "tree is essentially planar"
    # asymmetric: no node has a uniform full k-ary child count structure
    kids = {}
    for b in geom["branches"]:
        kids.setdefault(b["parent"], []).append(b["id"])
    counts = {len(v) for k, v in kids.items() if k is not None}
    assert len(counts) > 1, "every node has the same number of children"


# ---------------------------------------------------------------- camera ----


def test_problem_json_camera_schema(tmp_path):
    out = _generate(tmp_path)
    prob = json.loads((out / "problem" / "problem.json").read_text())
    assert prob["schema"] == "lumen-registration/1"
    assert prob["case_id"] == "full-tree-v1"
    assert prob["provenance"] == "procedural"
    assert prob["geometry"] == "geometry.json"
    assert prob["image"] == "image.png"
    cam = prob["camera"]
    for k in (
        "fx_px",
        "fy_px",
        "cx_px",
        "cy_px",
        "width_px",
        "height_px",
        "axes",
        "source_to_detector_mm",
    ):
        assert k in cam, k
    assert cam["width_px"] > 0 and cam["height_px"] > 0
    assert cam["fx_px"] > 0 and cam["source_to_detector_mm"] > 0


def test_projection_matches_true_carm(tmp_path):
    """Screen pixels must equal CArm.project + the nv-1-v row flip, in mm."""
    out = _generate(tmp_path)
    prob = json.loads((out / "problem" / "problem.json").read_text())
    pose = np.asarray(
        json.loads((out / "answer" / "pose.json").read_text())["model_to_camera_mm"], float
    )
    geom = json.loads((out / "problem" / "geometry.json").read_text())
    cam = prob["camera"]
    carm = CArm(
        source=np.zeros(3),
        detector_center=np.array([0.0, 0.0, cam["source_to_detector_mm"]]),
        up=np.array([0.0, -1.0, 0.0]),
        width=180.0,
        height=180.0,
        nu=cam["width_px"],
        nv=cam["height_px"],
    )
    b = geom["branches"][0]
    p_model = np.asarray(b["points_mm"], float)[:: max(1, len(b["points_mm"]) // 5)]
    p_cam = (pose[:3, :3] @ p_model.T).T + pose[:3, 3]
    for X, Y, Z in p_cam:
        u, v = carm.project([X, Y, Z])
        assert abs(cam["fx_px"] * X / Z + cam["cx_px"] - u) < 1e-6
        assert abs(cam["fy_px"] * Y / Z + cam["cy_px"] - (cam["height_px"] - 1 - v)) < 1e-6


def test_pose_is_proper_se3(tmp_path):
    out = _generate(tmp_path)
    pose = np.asarray(
        json.loads((out / "answer" / "pose.json").read_text())["model_to_camera_mm"], float
    )
    assert pose.shape == (4, 4)
    R, t = pose[:3, :3], pose[:3, 3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    assert np.allclose(pose[3], [0, 0, 0, 1])
    assert np.linalg.norm(t) > 0  # tree sits in front of the source
    assert json.loads((out / "answer" / "pose.json").read_text())["schema"] == "lumen-rigid-pose/1"


def test_geometry_stored_in_model_space(tmp_path):
    """Problem geometry must be pre-pose (camera space would trivially align)."""
    out = _generate(tmp_path)
    pose = np.asarray(
        json.loads((out / "answer" / "pose.json").read_text())["model_to_camera_mm"], float
    )
    geom = json.loads((out / "problem" / "geometry.json").read_text())
    root = next(b for b in geom["branches"] if b["parent"] is None)
    p0 = np.asarray(root["points_mm"][0], float)
    p_cam = pose[:3, :3] @ p0 + pose[:3, 3]
    assert not np.allclose(p0, p_cam, atol=1e-6)


# ----------------------------------------------------------------- image ----


def test_image_is_plain_grayscale_and_legible(tmp_path):
    out = _generate(tmp_path)
    img = gen.read_png(out / "problem" / "image.png")
    assert img.ndim == 2 and img.dtype == np.uint8
    assert img.shape[0] >= 256 and img.shape[1] >= 256
    assert int(img.std()) > 10, "image is flat"
    bg = float(np.median(img))
    assert bg > 150, "background is not bright"
    dark = img < bg - 60
    assert img.min() < bg - 100, "no strongly attenuated vessel pixels"
    frac = float(dark.mean())
    assert 0.005 < frac < 0.5


def test_answer_pose_not_leaked_into_problem(tmp_path):
    out = _generate(tmp_path)
    problem = out / "problem"
    assert {p.name for p in problem.iterdir()} == REQUIRED_PROBLEM
    manifest = json.loads((problem / "problem.json").read_text())
    assert set(manifest) == {"schema", "case_id", "provenance", "camera", "geometry", "image"}
    assert set(manifest["camera"]) == {
        "fx_px",
        "fy_px",
        "cx_px",
        "cy_px",
        "width_px",
        "height_px",
        "axes",
        "source_to_detector_mm",
    }
    geometry = json.loads((problem / "geometry.json").read_text())
    assert set(geometry) == {"branches"}
    for branch in geometry["branches"]:
        assert set(branch) == {"id", "parent", "territory", "points_mm", "radius_mm"}
    assert not (out / "problem" / "pose.json").exists()
    assert not (out / "problem" / "overlay.png").exists()


# --------------------------------------------------------------- overlay ----


def test_overlay_and_territory_masks(tmp_path):
    out = _generate(tmp_path)
    ov = gen.read_png(out / "answer" / "overlay.png")
    img = gen.read_png(out / "problem" / "image.png")
    assert ov.ndim == 3 and ov.shape[:2] == img.shape[:2]
    masks = json.loads((out / "answer" / "territories.json").read_text())
    total = np.zeros(img.shape, bool)
    for fname in masks.values():
        m = gen.read_png(out / "answer" / fname)
        assert m.ndim == 2 and m.shape == img.shape
        assert m.dtype == np.uint8
        assert m.sum() > 0
        total |= m > 0
    # every vessel-support pixel is claimed by exactly one territory mask
    support = img < 200  # attenuated (darker) pixels
    overlap = int((support & ~total).sum())
    assert overlap < 0.05 * int(support.sum()), (
        f"{overlap} support pixels unclaimed by territory masks"
    )


# ----------------------------------------------------------- determinism ----


def test_deterministic_regeneration_byte_identical(tmp_path):
    a = _generate(tmp_path)
    b = gen.generate_case(tmp_path / "case2")
    files = lambda d: {p.relative_to(d): p.read_bytes() for p in d.rglob("*") if p.is_file()}
    assert files(a) == files(b)


def test_cli_generates_case(tmp_path):
    out = tmp_path / "cli-case"
    r = subprocess.run(
        [sys.executable, "-m", "lumen.registration_pack.generate", str(out)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "problem" / "problem.json").is_file()


def test_box_downsample_preserves_native_principal_point():
    # An axial segment through the principal ray is radially symmetric on the detector.
    pose = np.eye(4)
    pose[2, 3] = 300
    branch = dict(points_mm=[[0, 0, 0], [0, 0, 2]], radius_mm=1.5, territory="inlet")
    image, _, _ = gen.render([branch], pose)
    darkness = (255 - image).astype(float)
    row, col = np.indices(image.shape)
    for axis in [col, row]:
        assert abs((axis * darkness).sum() / darkness.sum() - 255.5) < 1e-10
