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


def test_wall_deflection_depends_on_hgo_stiffness():
    # The constitutive claim — under the SAME contact load a softer HGO wall yields
    # more than a stiffer one — tested directly on the wall field. Deterministic and
    # platform-independent (one quasi-static solve; no chaotic sim trajectory).
    pytest.importorskip("warp")
    from lumen.newton.hgo_wall import WallField

    n_s, n_th = 20, 8
    load = np.zeros(n_s * n_th, np.float32)
    load[(n_s // 2) * n_th:(n_s // 2 + 1) * n_th] = 50.0   # a localized contact-load ring

    def deflect(params):
        w = WallField(R0=2.0, s_max=80.0, n_s=n_s, n_th=n_th, params=params, device="cpu")
        w.wall_load.assign(load)
        w.update_from_load()
        return w.max_deflection()

    soft = deflect(HGOParams(C10=2e3, k1=1e3, k2=1.0, thickness=0.3))
    stiff = deflect(HGOParams(C10=2e4, k1=1e4, k2=1.0, thickness=0.3))
    assert soft > 1e-3
    assert soft > stiff                          # softer wall deflects more


def test_deformable_wall_couples_in_sim_and_stays_stable():
    # The deformable wall actually deflects in the coupled Newton sim (and a rigid
    # wall never does), under sustained gentle contact — a stable equilibrium, not
    # the chaotic preload ram that made the old assertion platform-fragile.
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from lumen.newton.sim import NewtonGuidewireSim

    M, L, R, n = 40, 80.0, 2.0, 11
    vessel = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    dev = np.stack([np.full(n, 1.85), np.zeros(n),     # starts already in the contact band
                    np.linspace(4, 4 + 2.0 * (n - 1), n)], axis=1)

    def run(deformable, params=None):
        sim = NewtonGuidewireSim(vessel, R, dev, radius=0.2, stretch_stiffness=1e4,
                                 bend_stiffness=4e1, kappa=3e3, d_hat=0.3,
                                 deformable_wall=deformable, hgo_params=params,
                                 vbd_iterations=12, device="cpu")
        for _ in range(40):
            sim.step(dt=2.5e-2, substeps=5, preload=(40.0, 0.0, 0.0))
        return sim.wall_max_deflection(), np.isfinite(sim.node_radii()).all()

    rigid_defl, rigid_ok = run(deformable=False)
    soft_defl, soft_ok = run(True, HGOParams(C10=2e3, k1=1e3, k2=1.0, thickness=0.3))

    assert rigid_defl == 0.0 and rigid_ok        # rigid wall does not deflect; stable
    assert soft_ok and soft_defl > 1e-3          # deformable wall deflects; stays finite
