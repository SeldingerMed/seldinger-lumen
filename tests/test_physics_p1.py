"""P1 / doc M0: rod in a tube, tube-intrinsic contact + barrier, friction sysID.

torch is an optional extra; skip cleanly if it isn't installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lumen.core.frame import CenterlineFrame
from lumen.core.lumen_field import LumenField
from lumen.physics.contact import ContactGeometry, ContactParams
from lumen.physics.rod import Rod, RodParams
from lumen.physics.solver import SimConfig, Solver
from lumen.physics import sysid


def _straight_geom(L=80.0, R=2.0, M=40):
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    return ContactGeometry(cl, LumenField.cylinder(length=L, radius=R, n=2))


def test_torch_projection_matches_numpy_frame():
    # the batched torch projection must agree with the numpy reference frame
    cl = np.stack([np.zeros(20), np.zeros(20), np.linspace(0, 50, 20)], axis=1)
    geom = ContactGeometry(cl, LumenField.cylinder(50, 2.0, n=2))
    f = CenterlineFrame(cl)
    pts = np.array([[1.0, 0.0, 12.0], [0.0, 1.5, 33.0]])
    proj = geom.project(torch.tensor(pts, dtype=torch.float64).unsqueeze(0))
    for i, p in enumerate(pts):
        ref = f.project(p)
        assert abs(float(proj["s"][0, i]) - ref.s) < 1e-6
        assert abs(float(proj["r"][0, i]) - ref.r) < 1e-6
        assert abs(float(proj["theta"][0, i]) - ref.theta) < 1e-6


def test_barrier_pushes_penetrating_node_back():
    geom = _straight_geom()
    cp = ContactParams(kappa=1.5e3, d_hat=0.25)
    # one node penetrating the wall (r=2.1 > R=2.0): barrier force must point inward (-x)
    x = torch.tensor([[[2.1, 0.0, 40.0]]], dtype=torch.float64, requires_grad=True)
    E = geom.barrier_energy(x, cp)
    (g,) = torch.autograd.grad(E.sum(), x)
    force = -g
    assert force[0, 0, 0] < 0          # pushed back toward the lumen centre
    # a node well inside (r=0.5) feels no contact force
    x2 = torch.tensor([[[0.5, 0.0, 40.0]]], dtype=torch.float64)
    E2 = geom.barrier_energy(x2, cp)
    assert float(E2.detach()) == 0.0


def test_friction_reduces_tip_advance_monotonically():
    geom, make_rod, cfg, _ = sysid.sliding_experiment(0.0, steps=150)
    last = None
    advances = []
    for mu in [0.0, 0.3, 0.6, 1.0]:
        s = Solver(geom, contact=ContactParams(**sysid.CP), cfg=cfg)
        with torch.no_grad():
            x = s.rollout(make_rod(), mu=torch.tensor([mu], dtype=torch.float64)).x
        tip_s = float(geom.project(x)["s"][0, -1])
        advances.append(tip_s)
    # strictly decreasing tip advance with increasing friction
    assert all(a > b for a, b in zip(advances, advances[1:])), advances


def test_recover_friction_by_gradient():
    # short differentiable horizon -> clean gradient -> exact recovery
    for mu_true in (0.3, 0.6):
        geom, make_rod, cfg, observed = sysid.sliding_experiment(mu_true, steps=35)
        mu_hat, loss = sysid.recover_friction(geom, make_rod, cfg, observed,
                                              mu_init=0.05)
        assert abs(mu_hat - mu_true) < 0.03, (mu_true, mu_hat, loss)
