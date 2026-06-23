"""L1.3 — image-based navigation. The perception PIPELINE (render -> detect tip ->
features) is pinned here (pure numpy, no Newton); a learnability check (needs Newton)
shows a policy trains on the image observation."""

import numpy as np
import pytest

from lumen.assets import procedural
from lumen.sensors import CArm, FluoroSensor
from lumen.sensors.perception import detect_device_tip, device_centroid


def _wire(z0, z1, n=20, x=0.0):
    return np.stack([np.full(n, x), np.zeros(n), np.linspace(z0, z1, n)], axis=1)


# ---- perception primitives ----------------------------------------------------
def test_tip_detection_matches_projected_tip():
    wire = _wire(-12, 12)
    sensor = FluoroSensor(mu_device=1.0, res=40, n_samples=120, nu=64, nv=64)
    carm = sensor.default_carm(wire, axis=(1, 0, 0))
    img, _ = sensor.render(wire, carm=carm)
    u, v, present = detect_device_tip(img, leading="max_v")
    assert present == 1.0
    pu, pv = carm.project(wire[np.argmax(wire[:, 2])])         # project the max-z (leading) node
    assert abs(u - pu) < 4 and abs(v - pv) < 4


def test_detect_tip_empty_and_nan_robust():
    assert detect_device_tip(np.zeros((16, 16)))[2] == 0.0     # nothing -> absent
    img = np.zeros((16, 16)); img[10, 8] = 5.0; img[0, 0] = np.nan
    u, v, present = detect_device_tip(img)                     # M2: a NaN pixel must not blind it
    assert present == 1.0 and (u, v) == (8.0, 10.0)


def test_detect_tip_rejects_bad_leading():
    with pytest.raises(ValueError):                            # L4: typo'd leading is an error
        detect_device_tip(np.ones((8, 8)), leading="maxv")


def test_device_centroid_weights_toward_the_device():
    img = np.zeros((20, 20)); img[5:8, 14:17] = 3.0           # a bright blob upper-right
    u, v = device_centroid(img)
    assert u > 12 and v > 4


def test_carm_project_center_and_behind_source():
    carm = CArm.looking_at([0, 0, 0], axis=(1, 0, 0), nu=64, nv=64)
    u, v = carm.project([0.0, 0.0, 0.0])                       # look-at target -> detector centre
    assert abs(u - 31.5) < 1.0 and abs(v - 31.5) < 1.0
    bu, bv = carm.project([-300.0, 0.0, 0.0])                  # L3: behind the source -> not in beam
    assert np.isnan(bu) and np.isnan(bv)


# ---- the H1 pin: the image observation actually carries the nav signal --------
def test_image_obs_progress_signal_is_on_detector_and_spans():
    # Replicates FluoroBatchedNav's obs math without the sim: a VESSEL-sized C-arm puts
    # both target and tip ON the detector, and the progress feature (target_v - tip_v)/nv
    # runs from large at the inlet to ~0 at the target. With the old seed-sized C-arm the
    # target projected off-detector and this signal was absent (H1).
    asset = procedural.straight_tube(80.0, 2.0)
    pts, _ = asset.edge_arrays(asset.edges[0])
    pts = np.asarray(pts)
    sensor = FluoroSensor(mu_device=1.0, res=40, n_samples=120, nu=64, nv=64)
    carm = sensor.default_carm(pts, axis=(1, 0, 0))           # vessel-sized (the H1 fix)
    target_z = 0.7 * 80.0
    _, tv = carm.project([0.0, 0.0, target_z])
    assert 0 <= tv < carm.nv                                  # target is ON the detector

    def progress(tip_z):
        img, _ = sensor.render(_wire(tip_z - 15, tip_z), carm=carm)
        _, v, present = detect_device_tip(img, leading="max_v")
        assert present == 1.0
        return (tv - v) / carm.nv

    assert progress(10.0) > 0.4                               # tip at inlet: far from target
    assert progress(target_z) < 0.15                          # tip at target: ~0 -> real signal


def test_reversed_vessel_leading_direction():
    # M1/L5: a vessel running toward -z must pick the inserted (min-v) tip, not max-v.
    sensor = FluoroSensor(mu_device=1.0, res=40, n_samples=120, nu=64, nv=64)
    wire = _wire(0, -24)                                       # tip at z=-24 (inserted end)
    carm = sensor.default_carm(wire, axis=(1, 0, 0))
    start_v = carm.project([0, 0, 0])[1]
    tip_v_true = carm.project([0, 0, -24])[1]
    leading = "max_v" if tip_v_true >= start_v else "min_v"   # the rule FluoroBatchedNav uses
    img, _ = sensor.render(wire, carm=carm)
    _, v, _ = detect_device_tip(img, leading=leading)
    assert abs(v - tip_v_true) < 4                            # detected the inserted tip


# ---- learnability (needs Newton) ---------------------------------------------
def test_image_observation_policy_trains():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.rl.cem import train_cem
    from lumen.rl.fluoro_nav import FluoroBatchedNav, fluoro_env_factory

    asset = procedural.straight_tube(80.0, 2.0)
    pts, lumen = asset.edge_arrays(asset.edges[0])
    R = float(np.asarray(lumen.R).mean())
    # the FluoroBatchedNav target is on-detector (H1) — sanity on the real env
    env = FluoroBatchedNav(np.asarray(pts), R, 2, FluoroSensor(res=20, nu=24, nv=24),
                           lumen_field=lumen, device="cpu")
    assert 0 <= env.target_uv[1] < env.carm.nv

    sensor = FluoroSensor(mu_device=1.0, res=20, n_samples=50, nu=24, nv=24)
    factory = fluoro_env_factory(sensor, view_axis=(1, 0, 0))
    best, hist = train_cem(np.asarray(pts), R, lumen_field=lumen, env_factory=factory,
                           warm_start=(2, -3.0), pop=12, iters=7, device="cpu", seed=0)
    # feasibility on the easy task: an image-obs policy trains and reaches the target.
    # (This monotonic insertion task is also solvable from the reward alone; the
    # perception SIGNAL is pinned separately above. Vision-necessity needs a harder task.)
    assert hist[-1]["success_rate"] >= 0.6
    assert len(best) == 5
