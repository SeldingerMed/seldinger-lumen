"""Layer 1 L1.1 (registration) + L1.2 (device-as-sensor). Pure numpy (no Newton):
the renderer is numpy and the wall-yield inverse uses the HGO closed form."""

import numpy as np

from lumen.sensors import FluoroSensor
from lumen.sensors.device_as_sensor import (device_on_wall, estimate_wall_stiffness,
                                            sensitivity, wall_yield)
from lumen.sensors.registration import apply_se3, register


# ---- L1.1: registration -------------------------------------------------------
def _wire():
    a = np.linspace(0, 1.2, 16)
    return np.stack([4 * np.sin(a), np.zeros(16), np.linspace(-10, 10, 16)], axis=1)


def test_apply_se3_identity_and_rigidity():
    w = _wire()
    assert np.allclose(apply_se3(w, np.zeros(6)), w)              # identity pose
    moved = apply_se3(w, [0, 5, 0, 0, 0, 0.5])                    # translate + rotate
    # rigid: pairwise distances preserved
    d0 = np.linalg.norm(w[1:] - w[:-1], axis=1)
    d1 = np.linalg.norm(moved[1:] - moved[:-1], axis=1)
    assert np.allclose(d0, d1, atol=1e-9)


def test_registration_recovers_in_plane_pose():
    w = _wire()
    sensor = FluoroSensor(mu_device=1.0, res=36, n_samples=100, nu=48, nv=48)
    carm = sensor.default_carm(w, axis=(1, 0, 0))                 # view +x; in-plane = y,z
    true = np.array([0.0, 3.0, -2.5, 0.20, 0.0, 0.0])            # in-plane t + roll about view
    target, _ = sensor.render(apply_se3(w, true), carm=carm)
    pose, hist = register(target, w, sensor, carm, iters=22)
    assert hist[-1] < 0.05 * hist[0]                             # image loss collapsed
    assert abs(pose[1] - 3.0) < 1.2 and abs(pose[2] + 2.5) < 1.2  # in-plane translation recovered
    # L5: depth (pose[0], along the view axis) is intentionally NOT asserted — mono
    # view is depth-ambiguous (perspective gives only weak magnification cues).


def test_rodrigues_at_pi():
    # L4: a 180° rotation about +x maps (0,1,0)->(0,-1,0) (axis-ambiguous but well-defined)
    w = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = apply_se3(w, [0, 0, 0, np.pi, 0, 0])                   # rotate about centroid (0,0.5,0)
    assert np.allclose(out[1], [0.0, 0.0, 0.0], atol=1e-6)       # the +y node flips to the centroid's other side
    assert np.allclose(out[0], [0.0, 1.0, 0.0], atol=1e-6)


# ---- L1.2: device-as-sensor ---------------------------------------------------
def test_wall_yield_monotonic_in_stiffness():
    from lumen.newton.hgo_wall import HGOParams
    ws = [wall_yield(300.0, 2.0, HGOParams(C10=c, k1=c * 0.5, k2=1.0, thickness=0.3))
          for c in (1e3, 4e3, 1.6e4)]                            # k2 pinned (not default-reliant)
    assert ws[0] > ws[1] > ws[2] > 0.0                           # softer wall yields more


def test_estimate_recovers_planted_stiffness():
    sensor = FluoroSensor(mu_device=1.0, res=36, n_samples=90, nu=44, nv=44)
    nodes = device_on_wall(4e3)
    cx = sensor.default_carm(nodes, axis=(1, 0, 0))
    cy = sensor.default_carm(nodes, axis=(0, 1, 0))
    true = 6.0e3
    targets = [sensor.render(device_on_wall(true), carm=c)[0] for c in (cx, cy)]
    est, hist = estimate_wall_stiffness(targets, sensor, [cx, cy], init_C10=2e3, iters=16)
    assert abs(est - true) / true < 0.08                        # recovered (soft/identifiable regime)


def test_identifiability_drops_with_stiffness_and_biplanar_beats_mono():
    sensor = FluoroSensor(mu_device=1.0, res=36, n_samples=90, nu=44, nv=44)
    nodes = device_on_wall(4e3)
    cx = sensor.default_carm(nodes, axis=(1, 0, 0))
    cy = sensor.default_carm(nodes, axis=(0, 1, 0))
    soft = sensitivity(1.5e3, sensor, [cx, cy], bulge_dir=(1, 0, 0))
    stiff = sensitivity(2.4e4, sensor, [cx, cy], bulge_dir=(1, 0, 0))
    assert soft > stiff                                          # stiff wall less identifiable (the gate)
    # bulge along +x is the depth axis of view-x -> mono_x near-blind, biplanar resolves it
    mono = sensitivity(6e3, sensor, cx, bulge_dir=(1, 0, 0))
    bi = sensitivity(6e3, sensor, [cx, cy], bulge_dir=(1, 0, 0))
    assert bi > 10.0 * mono


def test_mono_depth_ambiguous_fails_under_noise_but_biplanar_recovers():
    # M3: pin the core L1.2 claim operationally. With the displacement along a view's
    # DEPTH axis the mono signal is ~1e-11 (perspective only) — noise-free it recovers
    # by that tiny clean signal, but under realistic image noise it's swamped and FAILS,
    # while the orthogonal second view (biplanar) keeps a strong in-plane signal.
    sensor = FluoroSensor(mu_device=1.0, res=36, n_samples=90, nu=44, nv=44)
    nodes = device_on_wall(4e3)
    cx = sensor.default_carm(nodes, axis=(1, 0, 0))            # bulge +x is depth for view +x
    cy = sensor.default_carm(nodes, axis=(0, 1, 0))            # ...and in-plane for view +y
    true, sigma = 8.0e3, 4e-3
    rng = np.random.default_rng(0)

    def noisy(carms):
        return [sensor.render(device_on_wall(true, bulge_dir=(1, 0, 0)), carm=c)[0]
                + rng.normal(0, sigma, (sensor.nv, sensor.nu)) for c in carms]

    mono, _ = estimate_wall_stiffness(noisy([cx]), sensor, [cx], init_C10=2e3, iters=14,
                                      bulge_dir=(1, 0, 0))
    bi, _ = estimate_wall_stiffness(noisy([cx, cy]), sensor, [cx, cy], init_C10=2e3, iters=14,
                                    bulge_dir=(1, 0, 0))
    assert abs(mono - true) / true > 0.3                       # under-determined -> fails
    assert abs(bi - true) / true < 0.15                        # biplanar resolves it
