"""Task #4: anisotropic fiber-aligned friction in the AVBD contact (doc §3.5.5).

Validated at the kernel level (the contact force the AVBD solve receives), which
isolates the friction model from chain dynamics. μ varies with the sliding
direction relative to the HGO collagen fiber direction.
"""

import numpy as np
import pytest

pytest.importorskip("warp")
import warp as wp

from lumen.newton.tube_barrier_kernel import accumulate_tube_barrier


def _friction_force(gamma_deg, mu_along=0.1, mu_across=0.8):
    """Run the contact kernel for one wall-penetrating body sliding +z; return force."""
    from lumen.core.frame import CenterlineFrame
    M = 10
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, 80, M)], axis=1)
    f = CenterlineFrame(cl)
    P = wp.array(f.points.astype(np.float32), dtype=wp.vec3)
    Tg = wp.array(f.tangents.astype(np.float32), dtype=wp.vec3)
    M1 = wp.array(f.m1.astype(np.float32), dtype=wp.vec3)
    cum_s = wp.array(f.cum_s.astype(np.float32), dtype=wp.float32)
    n_s, n_th = 4, 4
    r0 = wp.array(np.full(n_s * n_th, 2.0, dtype=np.float32), dtype=wp.float32)
    bq = wp.array(np.array([[1.8, 0, 40, 0, 0, 0, 1]], dtype=np.float32), dtype=wp.transform)
    bqd = wp.array(np.array([[0, 0, 0, 0, 0, 1.0]], dtype=np.float32), dtype=wp.spatial_vector)
    cg = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32)
    wm = wp.array(np.array([1], dtype=np.int32), dtype=wp.int32)
    wfield = wp.zeros(n_s * n_th, dtype=wp.float32)
    load = wp.zeros(n_s * n_th, dtype=wp.float32)
    bf = wp.zeros(1, dtype=wp.vec3)
    bh = wp.zeros(1, dtype=wp.mat33)
    wp.launch(accumulate_tube_barrier, dim=1,
              inputs=[cg, wm, bq, bqd, P, Tg, M1, cum_s, M, r0, float(f.length),
                      n_s, n_th, wfield, 2e3, 0.3, 0,
                      mu_along, mu_across, np.radians(gamma_deg)],
              outputs=[bf, bh, load])
    return bf.numpy()[0]


def test_anisotropic_friction_matches_coulomb_and_fiber_orientation():
    f_along = _friction_force(90.0)    # fibers axial -> slide is ALONG fibers -> mu_along
    f_across = _friction_force(0.0)    # fibers circumferential -> slide ACROSS -> mu_across
    fn = 2e3 * (0.3 - (2.0 - 1.8))     # kappa*(d_hat - dwall) = normal load = 200
    # normal (barrier) force points inward (-x), magnitude == fn
    assert abs(abs(f_along[0]) - fn) < 1e-2
    # friction (z) opposes the +z slide
    assert f_along[2] < 0 and f_across[2] < 0
    # Coulomb magnitude == mu * fn for each orientation
    assert abs(abs(f_along[2]) - 0.1 * fn) < 1e-1
    assert abs(abs(f_across[2]) - 0.8 * fn) < 1e-1
    # anisotropy: across-fiber sliding is grippier than along-fiber
    assert abs(f_across[2]) > 4.0 * abs(f_along[2])


def test_friction_off_by_default():
    f = _friction_force(40.0, mu_along=0.0, mu_across=0.0)
    assert abs(f[2]) < 1e-6            # no tangential force when friction disabled


def test_high_friction_full_sim_is_stable():
    # #19: the implicit friction Hessian keeps the coupled solve stable at high mu
    pytest.importorskip("newton")
    from lumen.newton.sim import NewtonGuidewireSim
    # device seated mid-vessel (away from the open ends) so it slides under axial
    # drag without exiting; tests that high-mu friction stays STABLE (no blow-up).
    M, L, R, n = 60, 120.0, 2.0, 11
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.full(n, 1.65), np.zeros(n), np.linspace(40, 40 + 2 * (n - 1), n)], axis=1)
    sim = NewtonGuidewireSim(vessel, R, dev, radius=0.2, kappa=3e3, d_hat=0.3,
                             mu_along=0.2, mu_across=0.9, gamma_fric_deg=30,
                             vbd_iterations=12, device="cpu")
    for _ in range(50):
        sim.step(dt=2.5e-2, substeps=5, preload=(120.0, 0.0, 25.0))
    r = sim.node_radii()
    assert np.isfinite(r).all() and r.max() < 3.5       # bounded — implicit friction didn't blow up
