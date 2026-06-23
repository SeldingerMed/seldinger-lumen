"""Layer 1 L1.0 — forward DRR fluoroscopy renderer (doc §4.1). Pure numpy."""

import numpy as np

from lumen.sensors import CArm, FluoroSensor


def _straight_wire(n=20, z0=-15, z1=15):
    return np.stack([np.zeros(n), np.zeros(n), np.linspace(z0, z1, n)], axis=1)


def test_carm_rays_are_unit_and_diverge_from_source():
    carm = CArm.looking_at([0, 0, 0], axis=(1, 0, 0), nu=16, nv=16)
    src, dirs = carm.rays()
    assert dirs.shape == (16, 16, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=2), 1.0, atol=1e-6)   # unit
    assert carm.pixel_points().shape == (16, 16, 3)


def test_straight_wire_projects_to_a_thin_line():
    wire = _straight_wire()
    A, carm = FluoroSensor(mu_device=1.0, res=64, n_samples=160).render(wire, radius=0.4)
    assert A.max() > 0.1
    lit = A > 0.5 * A.max()
    ys, xs = np.where(lit)
    x_span, y_span = xs.max() - xs.min(), ys.max() - ys.min()
    # a single straight wire -> a THIN LINE: long in one detector axis, narrow in the
    # other. Aspect ratio (not just "thin") would catch a focus-blurred blob too.
    assert x_span <= 8                                  # thin across
    assert y_span >= 0.6 * A.shape[0]                   # spans along
    assert y_span > 6 * (x_span + 1)                    # clearly a line, not a blob
    assert lit.mean() < 0.1                             # most of the field is empty


def test_single_node_device_renders_a_sphere_not_blank():
    # H1: a degenerate one-node device must still render (sphere), not silently blank
    A, _ = FluoroSensor(mu_device=1.0, res=48, n_samples=120).render(
        np.array([[0.0, 0.0, 0.0]]), radius=2.0)
    assert A.max() > 0.1
    import pytest
    with pytest.raises(ValueError):                     # but zero nodes is an error
        from lumen.sensors.volume import grid_for, voxelize_device
        g = grid_for(np.zeros((2, 3)), res=16)
        voxelize_device(np.zeros((0, 3)), 1.0, g)


def test_carm_up_parallel_to_view_falls_back_to_valid_axes():
    # L7: up ∥ view direction must still yield an orthonormal detector basis
    carm = CArm.looking_at([0, 0, 0], axis=(0, 0, 1), up=(0, 0, 1), nu=8, nv=8)
    u, v, n = carm.axes()
    for w in (u, v, n):
        assert abs(np.linalg.norm(w) - 1.0) < 1e-6
    assert abs(u @ v) < 1e-6 and abs(u @ n) < 1e-6 and abs(v @ n) < 1e-6


def test_degenerate_carm_raises():
    # L1: coincident source/detector has no view direction -> fail loud
    import pytest
    with pytest.raises(ValueError):
        CArm(source=np.zeros(3), detector_center=np.zeros(3)).axes()


def test_attenuation_scales_linearly_with_mu():
    wire = _straight_wire()
    _, carm = FluoroSensor(mu_device=1.0, res=64).render(wire, 0.4)
    a1, _ = FluoroSensor(mu_device=1.0, res=64, n_samples=160).render(wire, 0.4, carm=carm)
    a2, _ = FluoroSensor(mu_device=2.0, res=64, n_samples=160).render(wire, 0.4, carm=carm)
    assert abs(a2.max() / a1.max() - 2.0) < 0.05       # line integral linear in μ


def test_raycast_converges_in_samples():
    wire = _straight_wire()
    _, carm = FluoroSensor(res=64).render(wire, 0.4)
    a, _ = FluoroSensor(res=64, n_samples=160).render(wire, 0.4, carm=carm)
    b, _ = FluoroSensor(res=64, n_samples=320).render(wire, 0.4, carm=carm)
    assert abs(a.max() - b.max()) < 0.02 * b.max()


def test_radiograph_is_beer_lambert_dark_on_the_device():
    wire = _straight_wire()
    s = FluoroSensor(mu_device=1.5, res=64, n_samples=160)
    A, carm = s.render(wire, 0.4)
    I, _ = s.render(wire, 0.4, carm=carm, beer_lambert=True)
    assert I.max() <= 1.0 + 1e-6                        # I0 = 1
    assert I[np.unravel_index(A.argmax(), A.shape)] < 0.9   # dense device -> dark
