"""Generate case 3: the same selective injection seen from two calibrated sources."""

import argparse
from pathlib import Path

import numpy as np

from .generate import PROVENANCE, _camera_params, render, write_png
from .selective import _json

# Column vectors, mm. A and B look at the same rig isocenter (0, 0, 300) in A.
A_TO_B = np.array([[0.8, 0, 0.6, -180], [0, 1, 0, 0], [-0.6, 0, 0.8, 60], [0, 0, 0, 1]])
MODEL_TO_A = np.array([[0.96, -0.28, 0, 2], [0.28, 0.96, 0, -1], [0, 0, 1, 307], [0, 0, 0, 1]])


def _tree():
    branches = []
    t = np.linspace(0, 1, 25)[:, None]

    def add(parent, group, radius, controls):
        a, b, c, d = np.asarray(controls, dtype=float)
        points = (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t**2 * c + t**3 * d
        name = f"b{len(branches):02}"
        branches.append(
            dict(
                id=name,
                parent=parent,
                territory=group,
                radius_mm=radius,
                points_mm=points.round(4).tolist(),
            )
        )
        return name

    trunk = add(None, "inlet", 2.1, [(4, 31, 6), (6, 25, 4), (-1, 22, 5), (-3, 19, 8)])
    for territory, depth in [("west", -8), ("east", 8)]:
        feeder = add(
            trunk, "inlet", 1.6, [(-3, 19, 8), (-6, 18, 8), (-5, 18, depth), (-4, 15, depth)]
        )

        def curve(parent, radius, controls):
            return add(parent, territory, radius, np.asarray(controls) + [0, 0, depth])

        stem = curve(feeder, 1.5, [(-4, 15, 0), (-8, 10, 1), (5, 6, -2), (0, 1, 0)])
        curve(stem, 1.1, [(0, 1, 0), (-5, -4, 1), (-2, -9, 2), (-8, -14, 3)])
        curve(stem, 1.0, [(0, 1, 0), (5, -2, -1), (10, -6, -3), (9, -11, -2)])
    return branches


def generate_pack(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    branches = _tree()
    filled = [b for b in branches if b["territory"] == "west"]
    alternate = MODEL_TO_A.copy()
    alternate[2, 3] += 16
    for code, rig in [("a", np.eye(4)), ("b", A_TO_B)]:
        case_id = f"overlap-{code}-v1"
        problem, answer = (root / case_id / part for part in ("problem", "answer"))
        problem.mkdir(parents=True)
        answer.mkdir()
        _json(problem / "geometry.json", {"branches": branches})
        _json(
            problem / "problem.json",
            dict(
                schema="lumen-registration/1",
                case_id=case_id,
                provenance=PROVENANCE,
                image="image.png",
                geometry="geometry.json",
                camera=_camera_params(),
            ),
        )
        for name, matrix in [("pose", MODEL_TO_A), ("wrong_pose", alternate)]:
            _json(
                answer / f"{name}.json",
                dict(
                    schema="lumen-rigid-pose/1",
                    case_id=case_id,
                    model_to_camera_mm=(rig @ matrix).round(12).tolist(),
                ),
            )
        if code == "b":
            _json(
                problem / "calibration.json",
                dict(
                    schema="lumen-camera-pair/1",
                    from_case="overlap-a-v1",
                    to_case=case_id,
                    camera_a_to_camera_b_mm=rig.tolist(),
                ),
            )
        image, _, _ = render(filled, rig @ MODEL_TO_A)
        write_png(problem / "image.png", image)
        _json(
            answer / "visibility.json",
            dict(
                territory="west",
                visible=[b["id"] for b in filled],
                hidden=[b["id"] for b in branches if b["territory"] != "west"],
            ),
        )
        _json(answer / "territories.json", {"west": "mask.png", "east": "wrong_mask.png"})
        other, _, _ = render(filled, rig @ alternate)
        for label, mask in [("", image < 255), ("wrong_", other < 255)]:
            write_png(answer / f"{label}mask.png", mask.astype(np.uint8) * 255)
            p = np.pad(mask, 1)
            edge = mask & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
            overlay = np.repeat(image[:, :, None], 3, axis=2)
            overlay[edge] = (0, 120, 190)
            write_png(answer / f"{label}overlay.png", overlay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new directory for both views")
    generate_pack(parser.parse_args().destination)
