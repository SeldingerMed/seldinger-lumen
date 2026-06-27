"""Observation realism products useful to CV pipelines."""

import numpy as np

from lumen.assets import procedural
from lumen.core.frame import CenterlineFrame
from lumen.sensors import FluoroSensor, LuminalCamera
from lumen.sensors.preview import write_avi, write_png


def _scene():
    vessel = np.stack([np.zeros(24), np.zeros(24), np.linspace(-18, 18, 24)], axis=1)
    wire = np.stack([np.full(8, 0.3), np.zeros(8), np.linspace(-8, 8, 8)], axis=1)
    return vessel, wire


def test_fluoro_scene_returns_vessel_contrast_masks_and_keypoints():
    vessel, wire = _scene()
    sensor = FluoroSensor(res=28, nu=36, nv=36, n_samples=70)
    scene = sensor.render_scene(wire, contrast_nodes=vessel, contrast_radius=2.0,
                                mu_contrast=0.18)

    assert scene["image"].shape == (36, 36)
    assert scene["masks"]["device"].shape == (36, 36)
    assert scene["masks"]["vessel"].shape == (36, 36)
    assert scene["masks"]["device"].sum() > 0
    assert scene["masks"]["vessel"].sum() > scene["masks"]["device"].sum()
    assert scene["keypoints"]["tip"]["present"] is True
    u, v = scene["keypoints"]["tip"]["uv"]
    assert 0 <= u < 36 and 0 <= v < 36


def test_biplanar_fluoro_uses_two_distinct_calibrated_views():
    vessel, wire = _scene()
    sensor = FluoroSensor(res=24, nu=32, nv=32, n_samples=60)
    views = sensor.render_biplanar(wire, contrast_nodes=vessel, axes=((1, 0, 0), (0, 1, 0)))

    assert len(views) == 2
    assert all(v["image"].shape == (32, 32) for v in views)
    assert not np.allclose(views[0]["image"], views[1]["image"])
    assert views[0]["carm"].to_dict() != views[1]["carm"].to_dict()
    assert all(v["keypoints"]["tip"]["present"] for v in views)


def test_luminal_artifacts_are_seeded_and_visible():
    asset = procedural.straight_tube(80.0, 4.0)
    pts, lumen = asset.edge_arrays(asset.edges[0])
    pts = np.asarray(pts)
    frame = CenterlineFrame(pts)
    device = np.stack([pts[1], pts[4]])

    clean = LuminalCamera(nu=28, nv=28, n_steps=64).render(frame, lumen, device)
    art1 = LuminalCamera(nu=28, nv=28, n_steps=64, artifact_strength=0.35,
                         artifact_seed=3).render(frame, lumen, device)
    art2 = LuminalCamera(nu=28, nv=28, n_steps=64, artifact_strength=0.35,
                         artifact_seed=3).render(frame, lumen, device)

    assert np.array_equal(art1, art2)
    assert not np.allclose(clean, art1)
    assert art1.min() >= 0.0 and art1.max() <= 1.0


def test_png_and_video_preview_exports(tmp_path):
    frames = []
    for i in range(3):
        img = np.zeros((16, 16), float)
        img[4:12, 4 + i:7 + i] = 1.0
        frames.append(img)

    png = tmp_path / "preview.png"
    avi = tmp_path / "preview.avi"
    write_png(png, frames[0])
    write_avi(avi, frames, fps=5)

    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert avi.read_bytes()[:4] == b"RIFF"
    assert avi.stat().st_size > 1024


def test_preview_exports_create_parent_dirs_and_reject_mismatched_video_frames(tmp_path):
    import pytest

    nested = tmp_path / "previews" / "case0" / "frame.png"
    write_png(nested, np.ones((8, 8), float))
    assert nested.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError, match="same shape"):
        write_avi(tmp_path / "bad.avi", [np.zeros((8, 8)), np.zeros((8, 9))])
