"""L1.4 — fluoro realism seam + luminal RGB modality (pure numpy, no Newton)."""

import numpy as np
import pytest

from lumen.assets import procedural
from lumen.core.frame import CenterlineFrame
from lumen.sensors import FluoroSensor, LuminalCamera, RealismParams, degrade


# ---- realism seam ------------------------------------------------------------
def _block():
    A = np.zeros((48, 48)); A[20:28, 20:28] = 3.0      # bright device on a flat field
    return A


def test_default_realism_is_identity():
    A = _block()
    assert degrade(A) is A and np.array_equal(degrade(A), A)   # exact identity, returns input
    assert np.array_equal(degrade(A, RealismParams()), A)


def test_nonpositive_dose_rejected():
    for bad in (0.0, -3.0):
        with pytest.raises(ValueError):
            RealismParams(i0=bad)                              # else exp/log -> NaN/inf


def test_finite_dose_adds_noise_and_is_seed_reproducible():
    A = _block()
    n1 = degrade(A, RealismParams(i0=200.0, seed=7))
    n2 = degrade(A, RealismParams(i0=200.0, seed=7))
    assert np.array_equal(n1, n2)                       # seeded -> reproducible
    assert n1[:10, :10].std() > 1e-3                    # flat field is no longer flat
    assert degrade(A, RealismParams(i0=200.0, seed=8))[:10, :10].std() != n1[:10, :10].std()


def test_psf_spreads_edges_and_scatter_reduces_contrast():
    A = _block()
    blurred = degrade(A, RealismParams(psf_sigma=2.0))
    assert blurred.max() < A.max() and blurred[19, 24] > A[19, 24]
    assert np.ptp(degrade(A, RealismParams(scatter_frac=0.5))) < np.ptp(A)


def test_beam_hardening_is_concave_monotone():
    A = _block()
    h = degrade(A, RealismParams(beam_hardening=0.3))
    assert h.max() < A.max()                            # deep attenuation pulled down
    assert h[0, 0] == pytest.approx(0.0, abs=1e-9)      # zero path unchanged


def test_realism_threads_through_fluorosensor():
    sensor = FluoroSensor(res=24, nu=32, nv=32, n_samples=60)
    wire = np.stack([np.zeros(12), np.zeros(12), np.linspace(-10, 10, 12)], axis=1)
    carm = sensor.default_carm(wire, axis=(1, 0, 0))
    clean, _ = sensor.render(wire, carm=carm)
    noisy, _ = sensor.render(wire, carm=carm, realism=RealismParams(i0=500.0, seed=1))
    assert clean.shape == noisy.shape and not np.allclose(clean, noisy)


def test_fluoro_can_render_contrast_roadmap_context():
    sensor = FluoroSensor(res=28, nu=36, nv=36, n_samples=70)
    vessel = np.stack([np.zeros(24), np.zeros(24), np.linspace(-18, 18, 24)], axis=1)
    wire = np.stack([np.full(8, 0.3), np.zeros(8), np.linspace(-8, 8, 8)], axis=1)
    carm = sensor.default_carm(vessel, axis=(1, 0, 0))
    device_only, _ = sensor.render(wire, carm=carm)
    with_roadmap, _ = sensor.render(wire, carm=carm, contrast_nodes=vessel,
                                    contrast_radius=2.0, mu_contrast=0.18)
    assert with_roadmap.shape == device_only.shape
    assert with_roadmap.sum() > device_only.sum()          # vessel context contributes signal
    assert with_roadmap.max() >= device_only.max()         # device remains visible


# ---- luminal RGB modality ----------------------------------------------------
def _tip_setup(asset):
    pts, lumen = asset.edge_arrays(asset.edges[0])
    pts = np.asarray(pts)
    return CenterlineFrame(pts), lumen, np.stack([pts[1], pts[3]])  # forward-pointing device


def test_luminal_render_shape_and_range():
    frame, lumen, dev = _tip_setup(procedural.straight_tube(80.0, 4.0))
    img = LuminalCamera(nu=20, nv=20, n_steps=64).render(frame, lumen, dev)
    assert img.shape == (20, 20, 3) and img.min() >= 0.0 and img.max() <= 1.0


def test_luminal_shows_the_tunnel():
    # down a straight tube: off-axis rays hit the side wall nearer than the far end,
    # so the frame edge is brighter than the dead-ahead centre.
    frame, lumen, dev = _tip_setup(procedural.straight_tube(80.0, 4.0))
    gray = LuminalCamera(nu=24, nv=24, n_steps=64).render(frame, lumen, dev).mean(2)
    centre = gray[10:14, 10:14].mean()
    edge = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]]).mean()
    assert edge > centre


def test_luminal_stenosis_brightens_the_view():
    fw, lw, dw = _tip_setup(procedural.straight_tube(80.0, 4.0))
    fs, ls, ds = _tip_setup(procedural.stenotic_tube(80.0, 4.0, severity=0.7))
    wide = LuminalCamera(nu=20, nv=20, n_steps=64).render(fw, lw, dw)
    narrowed = LuminalCamera(nu=20, nv=20, n_steps=64).render(fs, ls, ds)
    assert narrowed.mean() > wide.mean()               # closer wall -> less falloff


def test_luminal_texture_adds_spatial_context():
    frame, lumen, dev = _tip_setup(procedural.straight_tube(80.0, 4.0))
    smooth = LuminalCamera(nu=24, nv=24, n_steps=64).render(frame, lumen, dev)
    textured = LuminalCamera(nu=24, nv=24, n_steps=64, texture_strength=0.18,
                             fold_strength=0.12).render(frame, lumen, dev)
    assert textured.shape == smooth.shape
    assert textured.std() > smooth.std()                # not a featureless radial gradient
    assert not np.allclose(textured, smooth)


def test_luminal_rejects_degenerate_device():
    frame, lumen, _ = _tip_setup(procedural.straight_tube(80.0, 4.0))
    with pytest.raises(ValueError):
        LuminalCamera().render(frame, lumen, np.zeros((1, 3)))   # need >= 2 nodes for a direction


def test_luminal_rejects_nonpositive_steps():
    with pytest.raises(ValueError):
        LuminalCamera(n_steps=0)                                 # else dtau = max_dist / 0


def test_sensor_swap_shares_one_scene():
    # the invariant: fluoro and luminal consume the SAME scene objects (frame points,
    # lumen field, device polyline) with no anatomy- or core-side change.
    asset = procedural.straight_tube(80.0, 4.0)
    frame, lumen, dev = _tip_setup(asset)
    rgb = LuminalCamera(nu=16, nv=16, n_steps=48).render(frame, lumen, dev)
    xray, _ = FluoroSensor(res=20, nu=24, nv=24, n_samples=50).render(dev)  # SAME device polyline
    assert rgb.ndim == 3 and xray.ndim == 2             # two modalities, one scene
