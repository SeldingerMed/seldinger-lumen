"""Task #3: HGO anisotropic hyperelastic wall sharing R (doc §3.4.2, §3.5.6).

The constitutive tests need only numpy; the coupled-wall test needs newton.
"""

import dataclasses

import numpy as np
import pytest

from lumen.newton.hgo_wall import HGOParams, hgo_psi, hgo_radial_stress


def test_hgo_constitutive_rest_state_and_analytic_derivative():
    p = HGOParams()
    assert abs(hgo_psi(1.0, p)) < 1e-9          # zero energy at rest
    assert abs(hgo_radial_stress(1.0, p)) < 1e-9
    # analytic stress == d(psi)/d(lambda) (validates the closed-form derivative)
    for lam in (1.05, 1.15, 1.30):
        h = 1e-6
        num = (hgo_psi(lam + h, p) - hgo_psi(lam - h, p)) / (2 * h)
        assert abs(hgo_radial_stress(lam, p) - num) / abs(num) < 1e-6


def test_hgo_fibers_engage_and_stiffen():
    p = HGOParams()
    # stress is monotone increasing and superlinear (exponential fiber stiffening)
    s = [hgo_radial_stress(l, p) for l in (1.05, 1.15, 1.30)]
    assert s[0] < s[1] < s[2]
    # fiber orientation matters (anisotropy): more circumferential fibers -> stiffer
    p_circ = dataclasses.replace(p, gamma_deg=10.0)
    assert hgo_radial_stress(1.2, p_circ) > hgo_radial_stress(1.2, p)


def test_deformable_wall_deflection_depends_on_hgo_stiffness():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.newton.sim import NewtonGuidewireSim

    M, L, R, n = 40, 80.0, 2.0, 11
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.full(n, 1.65), np.zeros(n),     # start near the wall
                    np.linspace(4, 4 + 2.0 * (n - 1), n)], axis=1)

    def run(deformable, params=None):
        sim = NewtonGuidewireSim(vessel, R, dev, radius=0.2, stretch_stiffness=1e4,
                                 bend_stiffness=4e1, kappa=3e3, d_hat=0.3,
                                 deformable_wall=deformable, hgo_params=params,
                                 vbd_iterations=12, device="cpu")
        for _ in range(120):
            sim.step(dt=5e-3, substeps=5, preload=(150.0, 0.0, 0.0))
        return sim.wall_max_deflection(), np.isfinite(sim.node_radii()).all()

    rigid_defl, _ = run(deformable=False)
    soft_defl, soft_ok = run(True, HGOParams(C10=2e3, k1=1e3, k2=1.0, thickness=0.3))
    stiff_defl, stiff_ok = run(True, HGOParams(C10=2e4, k1=1e4, k2=1.0, thickness=0.3))

    assert rigid_defl == 0.0                     # rigid wall does not deflect
    assert soft_ok and stiff_ok
    # softer HGO wall deflects more (a stiff wall may resist to ~0 deflection)
    assert soft_defl > 1e-2 and soft_defl > stiff_defl
