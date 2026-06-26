"""Accurate tier (doc §3.3): the IPC reference is penetration-free and cross-validates
the compliant fast tier. Pure-numpy (the fast-side comparison skips without newton)."""

import numpy as np

from lumen.accurate.ipc import IPCParams, IPCTubeReference, ipc_barrier


def test_ipc_barrier_blows_up_at_contact_and_vanishes_past_d_hat():
    d_hat, k = 0.3, 1.0e2
    b_far, bp_far = ipc_barrier(np.array([d_hat * 1.5]), d_hat, k)
    assert b_far[0] == 0.0 and bp_far[0] == 0.0          # inactive beyond d_hat
    # b -> +inf as d -> 0 (the penetration-free guarantee)
    b_near, _ = ipc_barrier(np.array([1e-3, 1e-6]), d_hat, k)
    assert b_near[1] > b_near[0] > 0.0
    # restoring: b'(d) < 0 in the band (force = -dE/dx pushes the node off the wall)
    _, bp = ipc_barrier(np.array([0.1]), d_hat, k)
    assert bp[0] < 0.0


def _straight():
    M, R, n = 40, 2.0, 11
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, 80, M)], axis=1)
    x0 = np.stack([np.full(n, 1.5), np.zeros(n), np.linspace(20, 40, n)], axis=1)
    return cl, R, x0


def test_ipc_reference_is_penetration_free_under_increasing_load():
    cl, R, x0 = _straight()
    ref = IPCTubeReference(cl, R, IPCParams(d_hat=0.3, kappa=1.0e2))
    prev_x = None
    for Fx in (50.0, 200.0, 800.0):
        x, info = ref.solve(x0, F=np.array([Fx, 0.0, 0.0]), iters=600)
        assert info["penetration_free"] and info["min_gap"] > 0.0   # never reaches the wall
        assert x[:, 0].max() < R                                     # every node stays inside
        if prev_x is not None:
            assert x[:, 0].max() >= prev_x - 1e-6                    # more load -> closer to wall
        prev_x = x[:, 0].max()


def test_crossval_accurate_is_penetration_free_where_fast_penetrates():
    from lumen.newton.crossval import crossval_penetration_free
    r = crossval_penetration_free(force=400.0)
    assert r["penetration_free"] and r["accurate_penetration"] == 0.0
    if r["fast_penetration"] is not None:               # newton present
        assert r["fast_penetration"] > 0.05             # the compliant fast tier penetrates
        assert r["accurate_penetration"] < r["fast_penetration"]


# ---- stochastic contact gradients (doc §3.5.8) -------------------------------
def test_deterministic_contact_gradient_is_dead_outside_the_band():
    from lumen.accurate.stochastic import contact_reaction, deterministic_grad
    react = lambda th: contact_reaction(th, R=2.0, d_hat=0.3, kappa=1.0)
    assert abs(deterministic_grad(react, 1.5)) < 1e-9     # inactive (gap 0.5 > d_hat) -> 0
    assert deterministic_grad(react, 1.9) > 0.1           # active (gap 0.1 < d_hat) -> alive


def test_smoothed_gradient_rehabilitates_and_recovers_from_the_flat_region():
    # §3.5.8: physical gap noise makes E[reaction] smooth, so the smoothed gradient is
    # nonzero even where the deterministic one is dead, and recovers the offset.
    from lumen.accurate.stochastic import contact_reaction, recover_by_smoothed_descent
    react = lambda th: contact_reaction(th, R=2.0, d_hat=0.3, kappa=1.0)
    theta_true, f_target = 1.85, float(contact_reaction(1.85, kappa=1.0))
    out = recover_by_smoothed_descent(f_target, 1.5, sigma=0.1, lr=0.5, iters=400, reaction=react)
    assert abs(out["det_grad0"]) < 1e-9                   # started where the raw gradient is dead
    assert abs(out["smooth_grad0"]) > 1e-3               # ...but the smoothed one is alive
    assert abs(out["theta"] - theta_true) < 0.05         # and it recovers the offset


def test_smoothing_sigma_trades_reach_for_bias():
    # the §3.5.8 identifiability knob: a larger physical-noise sigma reaches further into
    # the flat region but biases the smoothed optimum more. Pin that monotone tradeoff.
    from lumen.accurate.stochastic import contact_reaction, recover_by_smoothed_descent
    react = lambda th: contact_reaction(th, R=2.0, d_hat=0.3, kappa=1.0)
    theta_true, f_target = 1.85, float(contact_reaction(1.85, kappa=1.0))

    def bias(sigma):
        o = recover_by_smoothed_descent(f_target, 1.5, sigma=sigma, lr=0.5, iters=400, reaction=react)
        return abs(o["theta"] - theta_true)

    assert bias(0.2) > bias(0.1)                          # more smoothing -> more bias
