"""Independent checks on the two-view registration teaching fixture."""

import json
from pathlib import Path

import numpy as np
import pytest

from lumen.registration_pack import generate, score

PACK = Path(generate.__file__).parent


def load(path):
    return json.loads(path.read_bytes())


def xyz(branch, pose):
    return np.array(branch["points_mm"]) @ pose[:3, :3].T + pose[:3, 3]


def test_fork_association_changes_with_view(tmp_path):
    assert (PACK / "overlap.py").is_file(), "missing two-view generator"
    from lumen.registration_pack.overlap import generate_pack

    root = tmp_path / "pair"
    generate_pack(root)
    for view in ("a", "b"):
        case = root / f"overlap-{view}-v1"
        cam = load(case / "problem/problem.json")["camera"]
        branches = load(case / "problem/geometry.json")["branches"]
        pose = np.array(load(case / "answer/pose.json")["model_to_camera_mm"])
        wrong = np.array(load(case / "answer/wrong_pose.json")["model_to_camera_mm"])

        def group(t):
            return [b for b in branches if b["territory"] == t]

        def project(points):
            return points[:, :2] / points[:, 2:] * [cam["fx_px"], cam["fy_px"]] + [
                cam["cx_px"],
                cam["cy_px"],
            ]

        left, right = group("west"), group("east")
        assert len(left) == len(right) == 3
        true = np.concatenate([xyz(b, pose) for b in left])
        other = np.concatenate([xyz(b, pose) for b in right])
        np.testing.assert_allclose(np.linalg.norm(true - other, axis=1), 16)
        np.testing.assert_allclose(np.concatenate([xyz(b, wrong) for b in left]), other, atol=1e-9)
        rms = np.mean(np.sum((project(true) - project(other)) ** 2, axis=1)) ** 0.5
        im = generate.read_png(case / "problem/image.png")
        np.testing.assert_array_equal(im, generate.render(left, pose)[0])
        ref, alt = im < 255, generate.render(right, pose)[0] < 255
        dice = 2 * (ref & alt).sum() / (ref.sum() + alt.sum())
        if view == "a":
            assert rms <= 4 and dice >= 0.80
        else:
            assert rms >= 35 and dice <= 0.30
        assert score.score(case, case / "answer/wrong_pose.json")["rms_3d_mm"] == pytest.approx(16)
        assert score.score(case, case / "answer/pose.json")["rms_reprojection_px"] == 0


def test_staged_camera_contract(tmp_path):
    from lumen.registration_pack.overlap import generate_pack

    root = tmp_path / "pair"
    generate_pack(root)
    a, b = sorted(root.iterdir())
    files = {"image.png", "geometry.json", "problem.json"}
    assert {p.name for p in (a / "problem").iterdir()} == files
    assert {p.name for p in (b / "problem").iterdir()} == files | {"calibration.json"}
    rel = load(b / "problem/calibration.json")
    assert set(rel) == {"schema", "from_case", "to_case", "camera_a_to_camera_b_mm"}
    assert (rel["schema"], rel["from_case"], rel["to_case"]) == (
        "lumen-camera-pair/1",
        a.name,
        b.name,
    )
    matrix = np.array(rel["camera_a_to_camera_b_mm"])
    for item in ("pose", "wrong_pose"):

        def get_pose(case):
            return np.array(load(case / f"answer/{item}.json")["model_to_camera_mm"])

        np.testing.assert_allclose(matrix @ get_pose(a), get_pose(b), atol=1e-10)
    assert (a / "problem/geometry.json").read_bytes() == (b / "problem/geometry.json").read_bytes()
    for case in (a, b):
        manifest = load(case / "problem/problem.json")
        assert manifest == dict(
            schema="lumen-registration/1",
            case_id=case.name,
            image="image.png",
            geometry="geometry.json",
            provenance="procedural",
            camera=generate._camera_params(),
        )
        branches = load(case / "problem/geometry.json")["branches"]
        tips = {}
        for branch in branches:
            assert set(branch) == {"id", "parent", "radius_mm", "points_mm", "territory"}
            if branch["parent"] is not None:
                assert branch["points_mm"][0] == tips[branch["parent"]]
            assert branch["id"] not in tips
            tips[branch["id"]] = branch["points_mm"][-1]


def test_reveal_compares_forks_on_the_observed_pixels(tmp_path):
    from lumen.registration_pack.overlap import generate_pack

    root = tmp_path / "pair"
    generate_pack(root)
    for case in root.iterdir():
        answer = case / "answer"
        assert (answer / "territories.json").is_file(), "missing territory reveal"
        branches = load(case / "problem/geometry.json")["branches"]
        visible = load(answer / "visibility.json")
        assert visible == dict(
            territory="west",
            visible=[b["id"] for b in branches if b["territory"] == "west"],
            hidden=[b["id"] for b in branches if b["territory"] != "west"],
        )
        pose = np.array(load(answer / "pose.json")["model_to_camera_mm"])
        observed = generate.read_png(case / "problem/image.png")
        masks = load(answer / "territories.json")
        for group, overlay in [("west", "overlay.png"), ("east", "wrong_overlay.png")]:
            image, _, _ = generate.render([b for b in branches if b["territory"] == group], pose)
            mask = generate.read_png(answer / masks[group])
            np.testing.assert_array_equal(mask, (image < 255) * 255)
            diff = (generate.read_png(answer / overlay) != observed[:, :, None]).any(axis=2)
            assert diff.sum() > 100 and np.all(mask[diff] == 255)


def test_fresh_cli_matches_shipped_pair(tmp_path):
    import subprocess
    import sys

    target = tmp_path / "rebuild"
    cmd = [sys.executable, "-m", "lumen.registration_pack.overlap", str(target)]
    subprocess.run(cmd, check=True)
    assert target.is_dir(), "CLI must produce the pair"

    def files(root):
        return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}

    assert {p.name for p in target.iterdir()} == {"overlap-a-v1", "overlap-b-v1"}
    for case in target.iterdir():
        frozen = PACK / case.name
        assert files(case) == files(frozen)
        for rel in files(case):
            if rel.suffix == ".png":
                np.testing.assert_array_equal(
                    generate.read_png(case / rel), generate.read_png(frozen / rel)
                )
            else:
                assert (case / rel).read_bytes() == (frozen / rel).read_bytes()
    repeat = subprocess.run(cmd, capture_output=True, text=True)
    assert repeat.returncode and "FileExistsError:" in repeat.stderr
