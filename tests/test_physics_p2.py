"""P2 / doc M1: deformable shell sharing R; coupled rod-soft contact."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lumen.core.lumen_field import LumenField
from lumen.physics.contact import ContactGeometry, ContactParams
from lumen.physics.rod import Rod, RodParams
from lumen.physics.solver import SimConfig, Solver
from lumen.physics.wall import WallShell, WallShellParams


def _wall(k_axial=0.0, k_hoop=0.0, k_found=500.0, n_theta=24):
    return WallShell(np.linspace(0, 80, 40), R0_of_s=lambda s: 2.0, n_theta=n_theta,
                     params=WallShellParams(k_axial=k_axial, k_hoop=k_hoop,
                                            k_found=k_found))


def _equilibrate(wall, load_at, F, iters=30):
    w = wall.w.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        L = wall.energy(w).sum() - F * w[0, load_at[0], load_at[1]]
        L.backward()
        return L

    opt.step(closure)
    return w.detach()


def test_point_load_matches_analytic_winkler():
    # pure elastic foundation: deflection at the loaded node is exactly F / k_found
    wall = _wall(k_axial=0.0, k_hoop=0.0, k_found=500.0)
    w = _equilibrate(wall, (20, 12), F=120.0)
    assert abs(float(w[0, 20, 12]) - 120.0 / 500.0) < 1e-3


def test_anisotropy_spreads_along_stiffer_direction():
    # hoop much stiffer than axial -> a point load spreads more in theta (smaller
    # local drop) than in s
    wall = _wall(k_axial=50.0, k_hoop=800.0, k_found=300.0)
    w = _equilibrate(wall, (20, 12), F=200.0, iters=40)
    drop_axial = abs(float(w[0, 20, 12] - w[0, 18, 12]))
    drop_hoop = abs(float(w[0, 20, 12] - w[0, 20, 10]))
    assert drop_hoop < drop_axial


def test_shared_R_consistency():
    # the radius the contact barrier sees IS R0 + w (no separate collision mesh)
    M, L = 40, 80.0
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    geom = ContactGeometry(cl, LumenField.cylinder(L, 2.0, n=2))
    wall = _wall(k_found=300.0)
    wall.w = wall.w + 0.3                                   # uniform deflection
    solver = Solver(geom, wall=wall)
    x = torch.tensor([[[1.0, 0.0, 40.0]]], dtype=torch.float64)
    R_def = solver._deformed_R(x, wall.w)
    proj = geom.project(x)
    expected = geom._R_of_s(proj["s"]) + wall.sample(proj["s"], proj["theta"], wall.w)
    assert torch.allclose(R_def, expected)
    assert abs(float(R_def) - 2.3) < 1e-6                   # 2.0 nominal + 0.3 deflection


def test_softer_wall_deflects_more_under_load():
    M, L = 40, 80.0
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    geom = ContactGeometry(cl, LumenField.cylinder(L, 2.0, n=2))

    def make_rod():
        n, sp = 14, 2.0
        x0 = np.stack([np.full(n, 1.9), np.zeros(n),
                       np.linspace(2.0, 2.0 + sp * (n - 1), n)], axis=1)
        return Rod(torch.tensor(x0, dtype=torch.float64).unsqueeze(0),
                   RodParams(k_stretch=2e2, k_bend=2.0, damping=2e2))

    cfg = SimConfig(dt=8e-3, steps=80, anchor_base=True, insertion_rate=0.0,
                    preload_force=400.0)
    defl = []
    for kf in (2000.0, 600.0):
        wall = WallShell(geom.s_grid.numpy(), lambda s: 2.0, n_theta=24,
                         params=WallShellParams(k_axial=80.0, k_hoop=200.0,
                                                k_found=kf, drag=2e2))
        s = Solver(geom, contact=ContactParams(kappa=1.5e3, d_hat=0.25, mu=0.0),
                   cfg=cfg, wall=wall)
        with torch.no_grad():
            s.rollout(make_rod())
        defl.append(float(wall.w.max()))
    assert defl[1] > defl[0]                                # softer wall deflects more
