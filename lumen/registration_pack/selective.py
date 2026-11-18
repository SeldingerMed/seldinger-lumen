"""Rebuild the paired selective exposures of registration case 2 (NumPy, CPU)."""

import json
from pathlib import Path

import numpy as np

from .generate import PROVENANCE, _camera_params, build_tree, render, write_png

# Separate from the full-tree control. Right-handed rotation about camera X.
REFERENCE = [[1, 0, 0, 2], [0, 12 / 13, -5 / 13, -1], [0, 5 / 13, 12 / 13, 316], [0, 0, 0, 1]]


def _json(path, obj):
    path.write_bytes((json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n").encode())


def generate_pack(destination):
    """Write two independently scoreable cases into a fresh directory."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    branches = build_tree()
    for code, territory in [("a", "west"), ("b", "east")]:
        case_id = f"selective-{code}-v1"
        problem, answer = (root / case_id / part for part in ("problem", "answer"))
        problem.mkdir(parents=True)
        answer.mkdir()
        visible = [b for b in branches if b["territory"] == territory]
        pose = np.array(REFERENCE)
        image, _, masks = render(visible, pose)
        write_png(problem / "image.png", image)
        _json(problem / "geometry.json", {"branches": branches})
        _json(
            problem / "problem.json",
            dict(
                schema="lumen-registration/1",
                case_id=case_id,
                image="image.png",
                geometry="geometry.json",
                provenance=PROVENANCE,
                camera=_camera_params(),
            ),
        )
        _json(
            answer / "pose.json",
            dict(schema="lumen-rigid-pose/1", case_id=case_id, model_to_camera_mm=REFERENCE),
        )
        _json(
            answer / "visibility.json",
            dict(
                territory=territory,
                visible=[b["id"] for b in visible],
                hidden=[b["id"] for b in branches if b["territory"] != territory],
            ),
        )
        _json(answer / "territories.json", {territory: "mask.png"})
        write_png(answer / "mask.png", masks[territory].astype(np.uint8) * 255)
        # Move the entire rigid model along the source-to-filled-centroid ray.
        # Its projected centroid barely moves, but magnification changes.
        xyz = np.concatenate([b["points_mm"] for b in visible]) @ pose[:3, :3].T + pose[:3, 3]
        ray = xyz.mean(axis=0)
        wrong = pose.copy()
        wrong[:3, 3] = np.round(pose[:3, 3] + 12 * ray / np.linalg.norm(ray), 9)
        _json(
            answer / "wrong_pose.json",
            dict(schema="lumen-rigid-pose/1", case_id=case_id, model_to_camera_mm=wrong.tolist()),
        )
        wrong_image, _, _ = render(visible, wrong)
        write_png(answer / "wrong_mask.png", (wrong_image < 255).astype(np.uint8) * 255)
        projected = {}
        camera = _camera_params()
        for key, matrix, mask, filename in [
            ("reference", pose, image < 255, "overlay.png"),
            ("wrong", wrong, wrong_image < 255, "wrong_overlay.png"),
        ]:
            # Inner boundary, four-connected; the base is always the observed image.
            p = np.pad(mask, 1)
            edge = mask & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
            overlay = np.repeat(image[:, :, None], 3, axis=2)
            overlay[edge] = (0, 120, 190)
            write_png(answer / filename, overlay)
            projected[key] = {}
            for branch in branches:
                xyz = np.asarray(branch["points_mm"]) @ matrix[:3, :3].T + matrix[:3, 3]
                uv = xyz[:, :2] / xyz[:, 2:] * np.array(
                    [camera["fx_px"], camera["fy_px"]], float
                ) + np.array([camera["cx_px"], camera["cy_px"]], float)
                projected[key][branch["id"]] = np.round(uv, 5).tolist()
        _json(answer / "projections.json", projected)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new directory for exposures A and B")
    generate_pack(parser.parse_args().destination)
