"""P4 / doc M3: one-way flow coupling + generic clot/occlusion interface."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lumen.core.lumen_field import LumenField
from lumen.physics.contact import ContactGeometry, ContactParams
from lumen.physics.rod import Rod, RodParams
from lumen.physics.solver import SimConfig, Solver
from lumen.physics.flow import WindkesselFlow
from lumen.physics.occlusion import Occlusion


def _setup():
    M, L = 40, 80.0
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    geom = ContactGeometry(cl, LumenField.cylinder(L, 2.0, n=2))

    def make_rod():
        n, sp = 12, 2.0
        x0 = np.stack([np.zeros(n), np.zeros(n),
                       np.linspace(2.0, 2.0 + sp * (n - 1), n)], axis=1)
        return Rod(torch.tensor(x0, dtype=torch.float64).unsqueeze(0),
                   RodParams(k_stretch=2e2, k_bend=2.0, damping=2e2))

    return geom, make_rod


def test_windkessel_diastolic_decay_matches_analytic():
    wk = WindkesselFlow(R_periph=1.0, C=1.5)
    assert abs(wk.pressure_decay(100.0, 0.3) - 100 * math.exp(-0.3 / 1.5)) < 1e-9


def test_pulsatile_flow_peaks_above_mean():
    wk = WindkesselFlow(Q_mean=4.0, Q_pulse=2.0, heart_rate=1.0)
    peak = max(wk.Q(t * 0.01) for t in range(100))     # sweep one cycle
    assert peak > 4.0 and peak <= 6.0 + 1e-9


def test_flow_drag_advances_device_monotonically():
    geom, make_rod = _setup()
    cfg = SimConfig(dt=8e-3, steps=80, anchor_base=False)
    tips = []
    for q in (0.0, 3.0, 6.0):
        s = Solver(geom, contact=ContactParams(mu=0.0, kappa=1.5e3, d_hat=0.25),
                   cfg=cfg, flow=WindkesselFlow(Q_mean=q, Q_pulse=0.0, drag_coeff=30.0))
        with torch.no_grad():
            r = s.rollout(make_rod())
        tips.append(float(r.x[0, -1, 2]))
    assert tips[0] < tips[1] < tips[2]


def test_occlusion_engaged_mask_and_adhesion_restoring_force():
    occ = Occlusion(s_center=12.0, capture_radius=4.0, k_adhesion=3e2)
    s = torch.tensor([[2.0, 11.0, 13.0, 30.0]], dtype=torch.float64)
    mask = occ.engaged_mask(s)
    assert mask.tolist() == [[0.0, 1.0, 1.0, 0.0]]         # only nodes near s_center

    # adhesion resists pulling an engaged node away from its engagement position
    x_ref = torch.zeros(1, 4, 3, dtype=torch.float64)
    x = x_ref.clone()
    x[0, 1, 2] = 0.5                                       # displace engaged node 1
    x = x.requires_grad_(True)
    E = occ.adhesion_energy(x, x_ref, mask)
    assert float(E.detach()) > 0
    (g,) = torch.autograd.grad(E.sum(), x)
    assert float(-g[0, 1, 2]) < 0                          # restoring force points back


def test_coupled_flow_plus_occlusion_is_stable():
    geom, make_rod = _setup()
    occ = Occlusion(s_center=12.0, capture_radius=4.0, k_adhesion=3e2)
    s = Solver(geom, contact=ContactParams(mu=0.2, kappa=1.5e3, d_hat=0.25),
               cfg=SimConfig(dt=8e-3, steps=60, anchor_base=False, push_force=-300.0),
               flow=WindkesselFlow(Q_mean=4.0, Q_pulse=2.0), occlusion=occ)
    with torch.no_grad():
        r = s.rollout(make_rod())
    assert torch.isfinite(r.x).all()
    assert int(r.engaged.sum()) > 0                        # device engaged the clot
