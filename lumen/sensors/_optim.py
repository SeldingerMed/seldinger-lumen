"""Tiny finite-difference optimizer for the Layer-1 inverse loops.

The inverse problems here are low-dimensional (rigid pose ≈ 6, regional wall stiffness
1–few) over a SMOOTH forward (the soft-μ DRR renderer / the deformable-wall physics),
so a finite-difference gradient is a faithful, dependency-light way to invert them —
the "differentiable" loop the doc calls for (§3.6), realised numerically. The
single-graph Warp-autodiff port is a perf upgrade, not a correctness one, at these
dimensions. Optimises in per-parameter scaled coordinates so mixed units (mm vs rad)
are conditioned, with a backtracking line search.
"""

from __future__ import annotations

import numpy as np


def fd_minimize(f, x0, scale, iters=50, lr=0.4, h=1e-2, tol=1e-12, log=None):
    """Minimise f(x) from x0 by finite-difference gradient descent.

    `scale` (per-param) sets both the FD step (h·scale) and the natural step size, so
    translation (mm) and rotation (rad) advance commensurately. Returns (best_x, history)."""
    x = np.array(x0, dtype=float)
    s = np.asarray(scale, dtype=float)
    best_x, best_f = x.copy(), f(x)
    hist = [best_f]
    fails = 0
    for _ in range(iters):
        f0 = f(x)
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy(); xp[i] += h * s[i]
            g[i] = (f(xp) - f0) / h                  # ∂f/∂z_i in scaled coords (z = x/s)
        gn = np.linalg.norm(g)
        if gn < tol:
            break
        d = -g / gn                                   # unit descent direction (scaled)
        step, improved = lr, False
        for _ in range(25):                           # backtracking line search
            xt = x + step * s * d
            ft = f(xt)
            if ft < f0:
                x, improved = xt, True
                if ft < best_f:
                    best_f, best_x = ft, xt.copy()
                break
            step *= 0.5
        # M1: hist tracks the best loss so far (monotone) -> hist[-1] == f(best_x)
        hist.append(best_f)
        if log:
            log({"iter": len(hist) - 1, "loss": best_f, "grad": float(gn)})
        if not improved:
            # M2: a single failed line search shrinks the step and retries; only give
            # up after a few consecutive failures (a genuinely flat / sub-FD region).
            fails += 1
            lr *= 0.3
            if fails >= 3:
                break
        else:
            fails = 0
    return best_x, hist
