"""Accurate tier: a penetration-free quasi-static IPC contact reference (doc §3.3).

The fast tier (``lumen.newton``) is a *compliant* penalty barrier — fast and batched
for RL, but it allows a little penetration under load (penetration ≤ d_hat by design)
and is not penetration-free. The doc's accurate tier is a rigorous penetration-free
IPC solve used to cross-validate and calibrate the fast tier (§3.3, §3.8).

This is a small, self-contained reference (single instrument, not batched, not on the
RL hot path): a quasi-static energy minimisation of a discrete elastic rod against the
SAME tube-intrinsic wall R(s,θ), with the **IPC log barrier** (Li et al. 2020) and a
**feasibility-filtered line search** that never lets any node's wall gap reach zero —
so the equilibrium is provably penetration-free. It shares lumen.core.frame, so the
geometry matches the fast tier exactly. The heavy external oracles (STARK/SymX,
ppf-contact-solver) remain a drop-in via the same ``accurate_tier_status`` seam; this
gives an always-available reference without a GPU/C++ build.

Energy E(x) = ½ks·Σ(|eᵢ|−L0)²  (stretch)
            + ½kb·Σ|xᵢ₊₁−2xᵢ+xᵢ₋₁|²  (bending)
            + Σ b(dᵢ)               (IPC contact, dᵢ = R−rᵢ the wall gap)
            − Σ F·xᵢ                (external load)
with b(d) = −κ(d−d̂)²·ln(d/d̂) for 0<d<d̂, +∞ as d→0⁺ (the penetration-free guarantee).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lumen.core.frame import CenterlineFrame


def ipc_barrier(d, d_hat, kappa):
    """IPC log barrier b(d) and b'(d) on the active band 0<d<d_hat (else 0)."""
    d = np.asarray(d, dtype=float)
    active = (d > 0.0) & (d < d_hat)
    dd = np.where(active, np.clip(d, 1e-12, None), d_hat)
    diff = dd - d_hat
    ln = np.log(dd / d_hat)
    b = np.where(active, -kappa * diff * diff * ln, 0.0)
    bp = np.where(active, -kappa * (2.0 * diff * ln + diff * diff / dd), 0.0)
    return b, bp


@dataclass
class IPCParams:
    ks: float = 1.0e3        # stretch stiffness
    kb: float = 5.0e1        # bending stiffness
    kappa: float = 1.0e2     # IPC barrier stiffness
    d_hat: float = 0.3       # barrier activation distance (matches the fast tier)
    eps_gap: float = 1.0e-4  # feasibility floor: a step may never bring a gap below this


class IPCTubeReference:
    """Penetration-free quasi-static rod-in-tube solve (the accurate-tier reference)."""

    def __init__(self, centerline: np.ndarray, R, params: IPCParams | None = None):
        self.frame = CenterlineFrame(centerline)
        self.R = R                                   # scalar lumen radius (or callable s->R)
        self.p = params or IPCParams()

    def _R_at(self, s):
        return self.R(s) if callable(self.R) else float(self.R)

    def _gaps(self, x):
        """Per-node wall gap d=R−r and radial unit e_r (the contact geometry)."""
        d = np.empty(len(x))
        er = np.empty((len(x), 3))
        for i, xi in enumerate(x):
            pr = self.frame.project(xi)
            d[i] = self._R_at(pr.s) - pr.r
            er[i] = pr.e_r
        return d, er

    def energy_and_grad(self, x, L0, F):
        """Total energy and its gradient (node 0 is the fixed base; its grad is zeroed)."""
        p = self.p
        n = len(x)
        g = np.zeros((n, 3))
        E = 0.0
        # stretch
        e = x[1:] - x[:-1]
        L = np.linalg.norm(e, axis=1)
        E += 0.5 * p.ks * np.sum((L - L0) ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            dirv = np.where(L[:, None] > 0, e / L[:, None], 0.0)
        f = (p.ks * (L - L0))[:, None] * dirv
        g[:-1] -= f
        g[1:] += f
        # bending
        if n >= 3:
            lap = x[2:] - 2.0 * x[1:-1] + x[:-2]
            E += 0.5 * p.kb * np.sum(lap ** 2)
            g[:-2] += p.kb * lap
            g[1:-1] += -2.0 * p.kb * lap
            g[2:] += p.kb * lap
        # IPC contact
        d, er = self._gaps(x)
        b, bp = ipc_barrier(d, p.d_hat, p.kappa)
        E += float(np.sum(b))
        g += (-bp)[:, None] * er                     # dE/dx = b'(d)·(∂d/∂x), ∂d/∂x = -e_r
        # external load (free nodes)
        if F is not None:
            E -= float(np.sum(x[1:] @ F))
            g[1:] -= F
        g[0] = 0.0                                    # base is Dirichlet-fixed
        return E, g

    def solve(self, x_init, F=None, iters=400, tol=1e-4):
        """Minimise the energy to a penetration-free equilibrium (feasibility-filtered
        backtracking line search). Returns (x_eq, info)."""
        x = np.array(x_init, dtype=float)
        base = x[0].copy()
        L0 = np.linalg.norm(x[1:] - x[:-1], axis=1)   # rest lengths from the seed
        d0, _ = self._gaps(x)
        assert d0.min() > 0, "initial configuration must be penetration-free"
        E, g = self.energy_and_grad(x, L0, F)
        for it in range(iters):
            gn = float(np.linalg.norm(g[1:]))
            if gn < tol:
                break
            p_dir = -g
            alpha = 1.0
            ok = False
            for _ in range(60):                       # backtracking with a feasibility filter
                xt = x + alpha * p_dir
                xt[0] = base
                dt, _ = self._gaps(xt)
                if dt.min() > self.p.eps_gap:          # never penetrate / touch the wall
                    Et, _ = self.energy_and_grad(xt, L0, F)
                    if Et <= E + 1e-4 * alpha * float(np.sum(g * p_dir)):
                        x, E = xt, Et
                        ok = True
                        break
                alpha *= 0.5
            if not ok:
                break                                  # converged (no further feasible decrease)
            _, g = self.energy_and_grad(x, L0, F)
        d, _ = self._gaps(x)
        return x, {"min_gap": float(d.min()), "energy": float(E),
                   "grad_norm": float(np.linalg.norm(g[1:])), "iters": it,
                   "penetration_free": bool(d.min() > 0.0)}
