"""Differentiable path (doc §3.5.7): gradients through the constitutive model for
calibration. The fast tier (Newton VBD) is not autograd-differentiable by design;
this is the differentiable seam used to fit model parameters to observations.

It uses **Warp's autodiff** (wp.Tape) — no separate framework, same ecosystem as the
solver. The demonstrated inverse problem: recover HGO wall stiffness (C10, k1) from an
observed pressure–deflection curve by gradient descent. The same machinery extends to
the barrier, Ogden clot, and (via the accurate-tier equilibrium) the coupled solve.
"""

from __future__ import annotations

import numpy as np

try:
    import warp as wp
except Exception:  # pragma: no cover
    wp = None

if wp is not None:
    from lumen.newton.hgo_wall import _hgo_pressure_wp

    @wp.kernel
    def _hgo_curve_loss(theta: wp.array(dtype=wp.float32),     # [C10, k1] (requires_grad)
                        w_samples: wp.array(dtype=wp.float32),
                        p_target: wp.array(dtype=wp.float32),
                        R0: float, k2: float, kd: float, cg2: float, sg2: float,
                        thickness: float, loss: wp.array(dtype=wp.float32)):
        """Per-sample squared error between predicted and observed HGO shell pressure;
        differentiable in theta via the tape."""
        j = wp.tid()
        p = _hgo_pressure_wp(w_samples[j], R0, theta[0], theta[1], k2, kd, cg2, sg2, thickness)
        d = p - p_target[j]
        wp.atomic_add(loss, 0, d * d)


def hgo_pressure_curve(w_samples, C10, k1, R0=2.0, k2=1.0, kappa_d=0.1,
                       gamma_deg=40.0, thickness=0.3):
    """Forward HGO shell pressure at each deflection (numpy reference / target maker)."""
    from lumen.newton.hgo_wall import HGOParams, hgo_wall_pressure
    p = HGOParams(C10=C10, k1=k1, k2=k2, kappa_d=kappa_d, gamma_deg=gamma_deg,
                  thickness=thickness)
    return hgo_wall_pressure(np.asarray(w_samples, float), R0, p)


def calibrate_hgo(w_samples, p_target, init=(4.0e3, 2.0e3), R0=2.0, k2=1.0,
                  kappa_d=0.1, gamma_deg=40.0, thickness=0.3, lr=0.2, iters=400):
    """Recover (C10, k1) from an observed pressure–deflection curve by Warp-autodiff
    gradient descent. Optimises in NORMALISED parameter space (θ = z·init) so the two
    stiffnesses are well-conditioned. Returns (recovered (C10,k1), final loss, history)."""
    if wp is None:
        raise ImportError("warp required for the differentiable path")
    import math
    g = math.radians(gamma_deg)
    cg2, sg2 = math.cos(g) ** 2, math.sin(g) ** 2
    scale = np.asarray(init, np.float32)
    ws = wp.array(np.asarray(w_samples, np.float32), dtype=wp.float32)
    pt = wp.array(np.asarray(p_target, np.float32), dtype=wp.float32)
    z = np.ones(2, np.float32)                                   # normalised params, start at 1
    hist = []
    for it in range(iters):
        theta = wp.array((z * scale).astype(np.float32), dtype=wp.float32, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(_hgo_curve_loss, dim=len(w_samples),
                      inputs=[theta, ws, pt, float(R0), float(k2), float(kappa_d),
                              cg2, sg2, float(thickness)], outputs=[loss])
        tape.backward(loss)
        gz = theta.grad.numpy() * scale                         # chain rule: dL/dz = dL/dθ·scale
        gn = np.linalg.norm(gz)
        if gn > 0:
            z = z - lr * gz / gn                                 # normalised-gradient step
        hist.append(float(loss.numpy()[0]))
        tape.zero()
    final_loss = hist[-1] if hist else 0.0
    return (z * scale), final_loss, hist
