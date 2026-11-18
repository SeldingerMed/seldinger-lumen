import json
from pathlib import Path

import numpy as np
import pytest

from lumen.registration_pack import generate


PACK = Path(generate.__file__).parent


def load(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("variant,territory", [("a", "west"), ("b", "east")])
def test_selective_exposure_uses_only_named_branches(tmp_path, variant, territory):
    from lumen.registration_pack import selective

    root = tmp_path / "pack"
    selective.generate_pack(root)
    case = root / f"selective-{variant}-v1"
    problem, answer = case / "problem", case / "answer"
    manifest = load(problem / "problem.json")
    geometry = load(problem / "geometry.json")
    branches = generate.build_tree()
    assert geometry == {"branches": branches}  # Includes inactive anatomy, no fill labels.
    assert {p.name for p in problem.iterdir()} == {"image.png", "geometry.json", "problem.json"}
    assert set(manifest) == {"schema", "case_id", "geometry", "image", "camera", "provenance"}
    visible = [b for b in branches if b["territory"] == territory]
    hidden = [b for b in branches if b["territory"] != territory]
    reveal = load(answer / "visibility.json")
    assert reveal["visible"] == [b["id"] for b in visible]
    assert reveal["hidden"] == [b["id"] for b in hidden]
    assert reveal["territory"] == territory
    pose = np.array(load(answer / "pose.json")["model_to_camera_mm"])
    expected, _, masks = generate.render(visible, pose)
    observed = generate.read_png(problem / "image.png")
    np.testing.assert_array_equal(observed, expected)
    maskfile = load(answer / "territories.json")[territory]
    np.testing.assert_array_equal(generate.read_png(answer / maskfile), masks[territory] * 255)
    # A negative control: inactive geometry really would paint elsewhere.
    other, _, _ = generate.render(hidden, pose)
    assert np.count_nonzero((other < 255) & (observed == 255)) > 1000


@pytest.mark.parametrize("variant", ["a", "b"])
def test_misleading_rigid_overlay(tmp_path, variant):
    from lumen.registration_pack import selective, score

    root = tmp_path / "generated"
    selective.generate_pack(root)
    case = root / f"selective-{variant}-v1"
    problem, answer = case / "problem", case / "answer"
    observed = generate.read_png(problem / "image.png")
    foreground = observed < 255
    candidate = generate.read_png(answer / "wrong_mask.png") == 255
    dice = 2 * (candidate & foreground).sum() / (candidate.sum() + foreground.sum())
    assert 0.75 <= dice < 1
    assert score.score(case, answer / "wrong_pose.json")["rms_3d_mm"] == pytest.approx(12)
    assert score.score(case, answer / "pose.json")["rms_reprojection_px"] == 0
    # Check the delivered mask is actually the second 3D rendering, not a raster shift.
    branches = load(problem / "geometry.json")["branches"]
    t = load(answer / "visibility.json")["territory"]
    cam = load(problem / "problem.json")["camera"]
    for name, mask, overlay, key in [
        ("pose", foreground, "overlay", "reference"),
        ("wrong_pose", candidate, "wrong_overlay", "wrong"),
    ]:
        T = np.array(load(answer / (name + ".json"))["model_to_camera_mm"])
        active = [b for b in branches if b["territory"] == t]
        im, _, _ = generate.render(active, T)
        np.testing.assert_array_equal(im < 255, mask)
        rgb = generate.read_png(answer / (overlay + ".png"))
        marked = np.any(rgb != observed[..., None], axis=2)
        assert np.all(mask[marked]) and marked.sum() > 200
        # All branches, including hidden ones, retain model point order in projections.
        uv = load(answer / "projections.json")[key]
        assert set(uv) == {b["id"] for b in branches}
        for b in branches:
            xyz = np.array(b["points_mm"]) @ T[:3, :3].T + T[:3, 3]
            points = xyz[:, :2] / xyz[:, 2:] * [cam["fx_px"], cam["fy_px"]] + [
                cam["cx_px"],
                cam["cy_px"],
            ]
            np.testing.assert_allclose(uv[b["id"]], points, rtol=0, atol=1e-5)


def test_offline_rebuild_matches_both_frozen_exposures(tmp_path):
    import subprocess
    import sys

    root = tmp_path / "rebuilt"
    cmd = [sys.executable, "-m", "lumen.registration_pack.selective", str(root)]
    regenerated = subprocess.run(cmd, check=True, text=True, capture_output=True)
    assert str(root) in regenerated.stdout
    assert {p.name for p in root.iterdir()} == {"selective-a-v1", "selective-b-v1"}
    for case in root.iterdir():
        frozen = PACK / case.name
        files = {p.relative_to(case) for p in case.rglob("*") if p.is_file()}
        assert files == {p.relative_to(frozen) for p in frozen.rglob("*") if p.is_file()}
        for rel in files:
            if rel.suffix == ".png":
                np.testing.assert_array_equal(
                    generate.read_png(case / rel), generate.read_png(frozen / rel)
                )
            else:
                assert (case / rel).read_bytes() == (frozen / rel).read_bytes(), rel
    a, b = sorted(root.iterdir())
    for rel, key in [
        ("problem/problem.json", "camera"),
        ("answer/pose.json", "model_to_camera_mm"),
    ]:
        assert load(a / rel)[key] == load(b / rel)[key]
    assert (a / "problem/geometry.json").read_bytes() == (b / "problem/geometry.json").read_bytes()
    assert (a / "problem/image.png").read_bytes() != (b / "problem/image.png").read_bytes()
    repeat = subprocess.run(cmd, capture_output=True, text=True)
    assert repeat.returncode != 0 and "FileExistsError:" in repeat.stderr
