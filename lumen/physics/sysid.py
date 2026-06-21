"""System identification: recover a physical parameter by gradient (doc M0).

The honest use of differentiability (doc §3.5.7): gradients are reliable for
*calibration*, not necessarily for policy learning. Here we recover an unknown
friction coefficient by matching a simulated trajectory -- the gradient of the
trajectory-match loss w.r.t. mu flows through the (differentiable) friction force.

Run as a script for a self-contained demo:  python -m lumen.physics.sysid
"""

from __future__ import annotations

import numpy as np
import torch

from lumen.core.lumen_field import LumenField
from lumen.physics.contact import ContactGeometry, ContactParams
from lumen.physics.rod import Rod, RodParams
from lumen.physics.solver import SimConfig, Solver


CP = dict(kappa=1.5e3, d_hat=0.25)


def sliding_experiment(mu_true: float, steps: int = 35, dtype=torch.float64):
    """A rod pressed on a straight wall and dragged tangentially; friction sets
    how far the distal tip advances (capstan-style lag).

    A constant wall-ward preload (standing in for vessel curvature/pulsatility)
    sustains the normal load, and a compliant rod (low stretch stiffness) lets
    friction produce a clean, monotone tip lag. Returns (geom, make_rod, cfg,
    observed) where observed is the final rod state under mu_true; make_rod
    rebuilds the rod so the optimiser re-simulates from the same start.

    Horizon note (doc §3.5.7): the *forward* sim is stable for long rollouts, but
    backprop-through-time through stiff contact corrupts the gradient past ~50
    steps. A short horizon (default 35) keeps the calibration gradient clean and
    correctly signed -- "short differentiable rollouts only where helpful". Use a
    longer `steps` for forward-only effect demonstrations, not for gradients.
    """
    M, L = 40, 80.0
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, L, M)], axis=1)
    geom = ContactGeometry(cl, LumenField.cylinder(length=L, radius=2.0, n=2),
                           dtype=dtype)

    def make_rod():
        n, sp = 14, 2.0
        x0 = np.stack([np.full(n, 1.9), np.zeros(n),
                       np.linspace(2.0, 2.0 + sp * (n - 1), n)], axis=1)
        return Rod(torch.tensor(x0, dtype=dtype).unsqueeze(0),
                   RodParams(k_stretch=2.0e2, k_bend=2.0, damping=2.0e2))

    cfg = SimConfig(dt=8e-3, steps=steps, anchor_base=True, insertion_rate=0.05,
                    preload_force=350.0)
    solver = Solver(geom, contact=ContactParams(mu=mu_true, **CP), cfg=cfg)
    with torch.no_grad():
        observed = solver.rollout(make_rod()).x.clone()
    return geom, make_rod, cfg, observed


def recover_friction(geom, make_rod, cfg, observed, mu_init=0.05,
                     iters=20, lr=0.5, dtype=torch.float64):
    """Fit mu to match the observed final rod state. Returns (mu_hat, loss).

    LBFGS (line search) rather than Adam: the loss is a smooth, near-quadratic
    function of a single parameter, where Adam's unit-step normalisation wanders
    around the shallow minimum instead of settling into it.
    """
    mu = torch.tensor([mu_init], dtype=dtype, requires_grad=True)
    solver = Solver(geom, contact=ContactParams(**CP), cfg=cfg)
    opt = torch.optim.LBFGS([mu], lr=lr, max_iter=iters,
                            line_search_fn="strong_wolfe")
    last = {}

    def closure():
        opt.zero_grad()
        final = solver.rollout(make_rod(), mu=mu.clamp(1e-3, 2.0)).x
        loss = ((final - observed) ** 2).mean()
        loss.backward()
        last["loss"] = float(loss.detach())
        return loss

    opt.step(closure)
    return float(mu.detach().clamp(1e-3, 2.0)), last["loss"]


def main():
    mu_true = 0.45
    geom, make_rod, cfg, observed = sliding_experiment(mu_true)
    mu_hat, loss = recover_friction(geom, make_rod, cfg, observed, mu_init=0.05)
    print(f"mu_true={mu_true:.3f}  mu_hat={mu_hat:.3f}  loss={loss:.3e}")


if __name__ == "__main__":
    main()
