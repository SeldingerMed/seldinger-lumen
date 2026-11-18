import json
import struct
import zlib

import numpy as np

from lumen.registration_pack.generate import generate_case


def test_standard_png_stream(tmp_path):
    generate_case(tmp_path / "case")
    for path in (tmp_path / "case").rglob("*.png"):
        blob = path.read_bytes()
        start = blob.index(b"IDAT")
        length = struct.unpack(">I", blob[start - 4 : start])[0]
        assert zlib.decompress(blob[start + 4 : start + 4 + length])


def test_entire_tree_and_radius_inside_detector(tmp_path):
    case = generate_case(tmp_path / "case")

    def load(name):
        return json.loads((case / name).read_text())

    cam = load("problem/problem.json")["camera"]
    T = np.array(load("answer/pose.json")["model_to_camera_mm"])
    for branch in load("problem/geometry.json")["branches"]:
        p = np.array(branch["points_mm"]) @ T[:3, :3].T + T[:3, 3]
        uv = p[:, :2] / p[:, 2:] * [cam["fx_px"], cam["fy_px"]] + [cam["cx_px"], cam["cy_px"]]
        margin = cam["fx_px"] * branch["radius_mm"] / p[:, 2:]
        assert np.all(uv - margin > 0) and np.all(
            uv + margin < [cam["width_px"], cam["height_px"]]
        ), branch["id"]


def test_frozen_delivery_matches_generator(tmp_path):
    import importlib.resources

    frozen = importlib.resources.files("lumen.registration_pack") / "full-tree-v1"
    generated = generate_case(tmp_path / "new")
    for part in ("problem", "answer"):
        assert {p.name for p in (generated / part).iterdir()} == {
            p.name for p in (frozen / part).iterdir()
        }
        for p in (generated / part).iterdir():
            expected = (frozen / part / p.name).read_bytes()
            if p.suffix == ".png":
                # zlib encoder versions may produce different streams for identical scanlines.
                def pixels(blob):
                    start = blob.index(b"IDAT")
                    n = int.from_bytes(blob[start - 4 : start], "big")
                    return blob[: start - 4], zlib.decompress(blob[start + 4 : start + 4 + n])

                assert pixels(p.read_bytes()) == pixels(expected), p.name
            else:
                assert p.read_bytes() == expected, p.name
