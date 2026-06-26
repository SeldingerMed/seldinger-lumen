"""Physically-grounded stochastic contact gradients (doc §3.5.8).

The deterministic contact barrier is non-smooth: outside the active band (gap > d_hat)
the reaction is exactly 0, so its gradient w.r.t. a contact parameter is exactly 0 — a
calibrator started there has NO signal pointing toward contact, and gradient descent is
dead in the water. At the activation threshold the gradient kinks. These are the
pathologies §3.5.7 warns about.

§3.5.8's option — implemented and evaluated here, NOT asserted as the settled choice —
is to rehabilitate the gradient by modelling contact as a STOCHASTIC process whose noise
is tied to genuine physical uncertainty: the lubricating blood-film thickness, wall
roughness/heterogeneous friction, and device/wall manufacturing tolerance all jitter the
effective gap by ~σ. The *expected* reaction E_ξ[b(gap+ξ)] is then a smooth (Gaussian-
convolved) function of the parameter — nonzero even when the nominal gap is outside the
band, because the noise occasionally makes contact — so ∇E is well-defined everywhere and
recoverable by a randomized-smoothing (score-function) estimator. Crucially σ is a
PHYSICAL quantity, calibratable from data, not an arbitrary smoothing knob — which is the
whole point of §3.5.8 over ad-hoc barrier softening.

Honest scope: this is a sysID/calibration tool (§3.5.7 "use gradients for what they are
good for"), demonstrated on a contact-threshold toy. It is NOT a policy-gradient claim —
the doc defaults to model-free RL for control. The estimator is unbiased for the SMOOTHED
objective E_ξ[·]; whether that smoothed optimum is close enough to the true one is the
identifiability question the doc explicitly asks reviewers to judge, quantified by `sigma`.
"""

from __future__ import annotations

import numpy as np


def _scalar(v):
    """Coerce a scalar-valued fn's output to a python float (rejects non-scalar fns)."""
    return float(np.asarray(v).item())


def contact_reaction(theta, R=2.0, d_hat=0.3, kappa=2.0e3, mode="compliant"):
    """Deterministic wall-contact reaction at device radial position `theta` (gap = R−θ).

    Zero outside the active band (θ < R−d_hat), rising as the device presses in. This is
    the non-smooth observable whose gradient vanishes in the inactive region."""
    gap = R - np.asarray(theta, float)
    pen = d_hat - gap                                   # >0 inside the active band
    if mode == "compliant":
        return kappa * np.maximum(0.0, pen)
    # IPC log barrier magnitude (also flat-zero outside the band)
    dd = np.clip(gap, 1e-3, d_hat)
    diff = dd - d_hat
    bp = -kappa * (2.0 * diff * np.log(dd / d_hat) + diff ** 2 / dd)
    return np.where(gap < d_hat, np.abs(bp), 0.0)


def smoothed_value_and_grad(fn, theta, sigma, n_samples=256, rng=None):
    """Randomized-smoothing estimate of (E_ξ[fn(θ+ξ)], ∇_θ E_ξ[fn(θ+ξ)]) with ξ~N(0,σ²).

    Antithetic Gaussian (Stein/score-function) estimator — unbiased for the smoothed
    objective, with the antithetic pairing cancelling the odd-order variance:
        E[fn]   ≈ mean( (fn(θ+ξ)+fn(θ−ξ))/2 )
        ∇E[fn]  ≈ mean( ξ·(fn(θ+ξ)−fn(θ−ξ)) ) / (2σ²)
    `theta` may be a scalar or vector; `sigma` is the physical gap-uncertainty scale."""
    rng = rng or np.random.default_rng(0)
    th = np.atleast_1d(np.asarray(theta, float))
    xi = rng.normal(0.0, sigma, size=(n_samples, th.size))
    fp = np.array([_scalar(fn(th + x)) for x in xi])   # scalar-valued fn at θ+ξ
    fm = np.array([_scalar(fn(th - x)) for x in xi])   # antithetic θ−ξ
    value = float(np.mean((fp + fm) / 2.0))
    grad = np.mean(xi * (fp - fm)[:, None], axis=0) / (2.0 * sigma ** 2)
    return value, (grad if th.size > 1 else float(grad[0]))


def deterministic_grad(fn, theta, h=1e-4):
    """Central finite-difference gradient of the RAW (non-smoothed) observable — the
    thing that is 0 in the inactive region and kinks at the threshold."""
    th = np.atleast_1d(np.asarray(theta, float))
    g = np.zeros_like(th)
    for i in range(th.size):
        e = np.zeros_like(th); e[i] = h
        g[i] = (_scalar(fn(th + e)) - _scalar(fn(th - e))) / (2 * h)
    return g if th.size > 1 else float(g[0])


def recover_by_smoothed_descent(f_target, theta0, sigma, lr=2e-4, iters=200,
                                n_samples=256, reaction=None, rng=None):
    """Recover the device offset θ matching an observed reaction `f_target`, starting from
    `theta0` (which may sit in the FLAT inactive region where the deterministic gradient is
    0). Descends L(θ)=(E[reaction(θ)]−f_target)² using the smoothed gradient.

    Returns {theta, history, det_grad0, smooth_grad0}: the recovered θ plus the two
    gradients at the start, so a caller can see the deterministic one was ~0 (stuck) while
    the smoothed one gave a usable direction."""
    rng = rng or np.random.default_rng(0)
    reaction = reaction or contact_reaction
    theta = float(theta0)
    det_grad0 = deterministic_grad(reaction, theta)
    f0, sg0 = smoothed_value_and_grad(reaction, theta, sigma, n_samples, rng)
    smooth_grad0 = 2.0 * (f0 - f_target) * sg0          # ∇L via chain rule on the smoothed value
    history = []
    for _ in range(iters):
        f, g = smoothed_value_and_grad(reaction, theta, sigma, n_samples, rng)
        theta -= lr * 2.0 * (f - f_target) * g          # gradient step on L
        history.append((theta, f))
    return {"theta": theta, "history": history, "det_grad0": det_grad0, "smooth_grad0": smooth_grad0}


if __name__ == "__main__":  # self-check: the smoothed gradient rehabilitates the dead one
    R, d_hat = 2.0, 0.3
    react = lambda th: contact_reaction(th, R, d_hat, kappa=1.0)   # unit-scaled barrier
    theta_true = 1.85                                  # in contact (gap 0.15 < d_hat)
    f_target = _scalar(react(theta_true))
    theta0 = 1.5                                        # start OUTSIDE the band (gap 0.5 > d_hat)
    det0 = deterministic_grad(react, theta0)
    out = recover_by_smoothed_descent(f_target, theta0, sigma=0.1, lr=0.5, iters=400, reaction=react)
    assert abs(det0) < 1e-9, det0                       # deterministic gradient is dead at the start
    assert abs(out["smooth_grad0"]) > 1e-3              # smoothed gradient is alive
    assert abs(out["theta"] - theta_true) < 0.05, out["theta"]   # ...and it recovers θ
    print(f"stochastic-contact self-check ok: det grad0={det0:.2e} (dead), "
          f"smoothed recovered θ={out['theta']:.4f} vs {theta_true} (true)")
