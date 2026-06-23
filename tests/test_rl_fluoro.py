"""L1.3 — image-based navigation: a policy trained on FLUORO observations (device tip
detected in the rendered image, not privileged state) learns to navigate."""

import numpy as np
import pytest

from lumen.sensors import CArm, FluoroSensor
from lumen.sensors.perception import detect_device_tip


def _wire():
    return np.stack([np.zeros(20), np.zeros(20), np.linspace(-12, 12, 20)], axis=1)


def test_tip_detection_matches_projected_tip():
    # perception (detect_device_tip) agrees with geometry (CArm.project) for the tip node
    wire = _wire()
    sensor = FluoroSensor(mu_device=1.0, res=40, n_samples=120, nu=64, nv=64)
    carm = sensor.default_carm(wire, axis=(1, 0, 0))
    img, _ = sensor.render(wire, carm=carm)
    u, v, present = detect_device_tip(img)
    assert present == 1.0
    # the leading (max-v) device pixel should match the projection of the max-z node
    tip3d = wire[np.argmax(wire[:, 2])]
    pu, pv = carm.project(tip3d)
    assert abs(u - pu) < 4 and abs(v - pv) < 4               # within a few pixels


def test_detect_tip_empty_image_reports_absent():
    u, v, present = detect_device_tip(np.zeros((16, 16)))
    assert present == 0.0


def test_carm_project_roundtrips_to_pixel_grid_center():
    carm = CArm.looking_at([0, 0, 0], axis=(1, 0, 0), nu=64, nv=64)
    u, v = carm.project([0.0, 0.0, 0.0])                     # the look-at target -> detector centre
    assert abs(u - (64 / 2 - 0.5)) < 1.0 and abs(v - (64 / 2 - 0.5)) < 1.0


def test_image_observation_policy_learns():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.assets import procedural
    from lumen.rl.cem import train_cem
    from lumen.rl.fluoro_nav import fluoro_env_factory

    asset = procedural.straight_tube(80.0, 2.0)
    pts, lumen = asset.edge_arrays(asset.edges[0])
    sensor = FluoroSensor(mu_device=1.0, res=20, n_samples=50, nu=24, nv=24)
    factory = fluoro_env_factory(sensor, view_axis=(1, 0, 0))
    best, hist = train_cem(np.asarray(pts), float(np.asarray(lumen.R).mean()),
                           lumen_field=lumen, env_factory=factory, warm_start=(2, -3.0),
                           pop=12, iters=7, device="cpu", seed=0)
    assert hist[-1]["success_rate"] > hist[0]["success_rate"]   # learned from the image
    assert hist[-1]["success_rate"] >= 0.6                      # and reaches the target reliably
    assert len(best) == 5                                       # 4 image features + bias
