"""Warp backend: parity with the torch reference + autodiff correctness.

Runs on the Warp CPU device locally and on CUDA where present. Skipped if Warp
is not installed (it is an optional backend).
"""

import numpy as np
import pytest

warp = pytest.importorskip("warp")
torch = pytest.importorskip("torch")

from lumen.physics.warp_contact import WarpTubeContact
from lumen.physics.contact import ContactGeometry, ContactParams
from lumen.core.lumen_field import LumenField


def _scene(B=4, N=12, M=40, L=80.0, R=2.0, seed=0):
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    rng = np.random.default_rng(seed)
    x = np.stack([np.full(N, 1.0), np.zeros(N), np.linspace(4, L - 4, N)], axis=1)
    x = np.broadcast_to(x, (B, N, 3)).copy() + 0.6 * rng.standard_normal((B, N, 3))
    return cl, x, R, L


def test_warp_gap_and_energy_match_torch():
    cl, x, R, L = _scene()
    wc = WarpTubeContact(cl, R, device="cpu")
    gap_w, _ = wc.forces(x)
    e_w, _ = wc.barrier_energy_and_grad(x)

    geom = ContactGeometry(cl, LumenField.cylinder(L, R, n=2))
    xt = torch.tensor(x, dtype=torch.float64)
    proj = geom.project(xt)
    gap_t = (R - proj["r"]).numpy()
    e_t = float(geom.barrier_energy(xt, ContactParams(kappa=1.5e3, d_hat=0.25)).sum())

    assert np.abs(gap_w - gap_t).max() < 1e-4
    assert abs(e_w - e_t) / e_t < 1e-4


def test_warp_autodiff_force_equals_analytic_force():
    # the Warp tape gradient of the barrier energy must equal the analytic force
    cl, x, R, L = _scene(seed=1)
    wc = WarpTubeContact(cl, R, device="cpu")
    _, force = wc.forces(x)
    _, grad = wc.barrier_energy_and_grad(x)
    assert np.abs(force - (-grad)).max() < 1e-3


def test_warp_force_matches_torch_force():
    cl, x, R, L = _scene(seed=2)
    wc = WarpTubeContact(cl, R, device="cpu")
    _, force_w = wc.forces(x)
    geom = ContactGeometry(cl, LumenField.cylinder(L, R, n=2))
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    E = geom.barrier_energy(xt, ContactParams(kappa=1.5e3, d_hat=0.25))
    (g,) = torch.autograd.grad(E.sum(), xt)
    assert np.abs(force_w - (-g.numpy())).max() < 1e-2
