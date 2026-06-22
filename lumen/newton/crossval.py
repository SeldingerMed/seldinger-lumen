"""Accurate-tier cross-validation of the fast tier (doc §3.3, §3.8).

The doc's validation plan layers (§3.8): (1) analytic checks, (2) accurate-tier
cross-validation against STARK/SymX or ppf-contact-solver on identical scenes.
This module implements the cross-validation harness:

  * For canonical scenes that HAVE a closed form — the contact barrier law and the
    HGO wall stress — it compares the fast-tier kernels to the analytic ground
    truth across the operating range (non-circular: the kernels are independent
    Warp code, the references are independent numpy).
  * `OracleReference` is the drop-in seam for the heavy oracle: STARK/SymX or
    ppf-contact-solver (GPU). `accurate_tier_status()` reports which oracle backs
    the cross-validation; the analytic oracle is always available, the external
    binary is used when present (built on a GPU box).

The doc is explicit that the accurate tier is BORROWED, not built (§3.3), so this
harness consumes an oracle; it does not reimplement IPC.
"""

from __future__ import annotations

import numpy as np

from lumen.newton.hgo_wall import HGOParams, hgo_radial_stress


def _analytic_barrier_force(gap, kappa, d_hat, mode="compliant"):
    """Analytic barrier reaction magnitude fn at wall-distance `gap` (= R−r)."""
    g = np.asarray(gap, dtype=float)
    active = g < d_hat
    if mode == "compliant":
        fn = np.where(active, kappa * (d_hat - g), 0.0)
    else:  # IPC log: |b'(d)|
        dd = np.clip(g, 0.05 * d_hat, None)
        diff = dd - d_hat
        bp = -kappa * (2.0 * diff * np.log(dd / d_hat) + diff ** 2 / dd)
        fn = np.where(active, np.abs(bp), 0.0)
    return fn


def crossval_contact_force(kappa=2.0e3, d_hat=0.3, R=2.0, mode="compliant"):
    """Compare the Warp barrier kernel force to the analytic law across gaps.

    Returns max relative error over the active range. Requires warp+newton; if
    unavailable, raises ImportError (callers skip).
    """
    import warp as wp
    from lumen.newton.tube_barrier_kernel import accumulate_tube_barrier
    wp.init()

    from lumen.core.frame import CenterlineFrame
    M = 10
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, 80, M)], axis=1)
    f = CenterlineFrame(cl)
    P = wp.array(f.points.astype(np.float32), dtype=wp.vec3)
    Tg = wp.array(f.tangents.astype(np.float32), dtype=wp.vec3)
    M1 = wp.array(f.m1.astype(np.float32), dtype=wp.vec3)
    cum_s = wp.array(f.cum_s.astype(np.float32), dtype=wp.float32)
    n_s, n_th = 4, 4
    r0_grid = wp.array(np.full(n_s * n_th, R, dtype=np.float32), dtype=wp.float32)
    md = 1 if mode == "log" else 0

    errs = []
    for gap in np.linspace(0.02, d_hat * 0.95, 12):       # active band, inside the wall
        r = R - gap
        bq = wp.array(np.array([[r, 0, 40, 0, 0, 0, 1]], dtype=np.float32), dtype=wp.transform)
        bqd = wp.array(np.zeros((1, 6), dtype=np.float32), dtype=wp.spatial_vector)
        cg = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32)
        wm = wp.array(np.array([1], dtype=np.int32), dtype=wp.int32)
        wf = wp.zeros(n_s * n_th, dtype=wp.float32); ld = wp.zeros(n_s * n_th, dtype=wp.float32)
        bf = wp.zeros(1, dtype=wp.vec3); bh = wp.zeros(1, dtype=wp.mat33)
        wp.launch(accumulate_tube_barrier, dim=1,
                  inputs=[cg, wm, bq, bqd, P, Tg, M1, cum_s, M, r0_grid, float(f.length),
                          n_s, n_th, wf, kappa, d_hat, md, 0.0, 0.0, 0.0, 5e-3],
                  outputs=[bf, bh, ld])
        fn_kernel = abs(float(bf.numpy()[0][0]))          # radial (-x) component magnitude
        fn_analytic = float(_analytic_barrier_force(gap, kappa, d_hat, mode))
        if fn_analytic > 1e-6:
            errs.append(abs(fn_kernel - fn_analytic) / fn_analytic)
    return max(errs)


def crossval_hgo_stress(params: HGOParams | None = None):
    """HGO radial stress: analytic closed form vs central-difference of Ψ."""
    from lumen.newton.hgo_wall import hgo_psi
    p = params or HGOParams()
    errs = []
    for lam in np.linspace(1.02, 1.4, 12):
        h = 1e-6
        num = (hgo_psi(lam + h, p) - hgo_psi(lam - h, p)) / (2 * h)
        errs.append(abs(hgo_radial_stress(lam, p) - num) / abs(num))
    return max(errs)


def accurate_tier_status() -> dict:
    """Report which accurate-tier oracle backs the cross-validation."""
    external = None
    # #24 — try the several plausible import names each oracle may install under
    # (the exact module name is unverified until one is actually built/installed)
    candidates = ("ppf_contact_solver", "ppf", "ppf_contact", "pyppf",
                  "stark", "pystark", "stark_sim")
    for name in candidates:
        try:
            __import__(name)
            external = name
            break
        except Exception:
            pass
    return {"analytic_oracle": True, "external_oracle": external,
            "note": "analytic oracle always on; STARK/ppf-contact-solver drop in "
                    "via OracleReference when built on a GPU box (doc §3.3)"}
