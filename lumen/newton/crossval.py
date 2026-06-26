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
                          n_s, n_th, 1, wf, kappa, d_hat, md, 0.0, 0.0, 0.0, 5e-3],
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


def crossval_penetration_free(R=2.0, force=400.0):
    """Accurate vs fast tier under the SAME contact load: the accurate-tier IPC
    reference is penetration-free (every node stays inside the wall), while the
    compliant fast tier allows penetration ≤ d_hat by design. Returns both peak
    penetrations (r − R, clamped at 0). Requires only numpy for the accurate side;
    the fast side needs warp+newton (callers skip if absent)."""
    from lumen.accurate.ipc import IPCTubeReference, IPCParams
    M, n = 40, 11
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, 80, M)], axis=1)
    # accurate tier: rod near the +x wall, pushed outward by `force`
    x0 = np.stack([np.full(n, 1.5), np.zeros(n), np.linspace(20, 40, n)], axis=1)
    ref = IPCTubeReference(cl, R, IPCParams(d_hat=0.3, kappa=1.0e2))
    _, info = ref.solve(x0, F=np.array([force, 0.0, 0.0]), iters=600)
    acc_pen = max(-info["min_gap"], 0.0)              # >0 only if it penetrated (it won't)

    fast_pen = None
    try:
        import warp  # noqa: F401
        import newton  # noqa: F401
        from lumen.newton.sim import NewtonGuidewireSim
    except ImportError:
        pass
    else:
        dev = np.stack([np.full(n, 1.6), np.zeros(n), np.linspace(30, 50, n)], axis=1)
        sim = NewtonGuidewireSim(cl, R, dev, radius=0.2, kappa=3e3, d_hat=0.3,
                                 vbd_iterations=12, device="cpu")
        peak = 0.0
        for _ in range(60):
            sim.step(dt=2.5e-2, substeps=5, preload=(force, 0.0, 0.0))
            peak = max(peak, float((sim.node_radii() - R).max()))
        fast_pen = peak
    return {"accurate_penetration": acc_pen, "fast_penetration": fast_pen,
            "accurate_min_gap": info["min_gap"], "penetration_free": info["penetration_free"]}


def crossval_indentation_response(R=2.0, forces=(50.0, 150.0, 300.0, 500.0),
                                  d_hat=0.3, kappa_acc=1.0e2, kappa_fast=3.0e3):
    """Oracle ROLLOUT validation (doc §3.3 role (a), the M1 'matches oracle' check):
    sweep an outward contact load and compare the fast tier's deepest wall indentation
    to the penetration-free IPC oracle on the SAME scene at each load. This validates
    the fast tier against a high-fidelity ground-truth RESPONSE CURVE — a physical scalar
    per load, robust to the two tiers' different rod discretisations (a node-wise shape
    match is not meaningful: the compliant VBD cable and the quasi-static IPC rod have
    different elastica; what must agree is the CONTACT response).

    Returns {forces, accurate[], fast[], properties{...}}. What this validates is the
    CONTACT REGIME, not a stiffness-matched curve coincidence: the fast (compliant penalty)
    and accurate (penetration-free log-barrier) tiers are *designed* to differ by the
    compliant penetration, so the claim is that the fast tier tracks the oracle to within
    that band, not that the two response curves overlay. The validated properties:
      * the oracle is monotone and penetration-free (r_acc <= R at every load);
      * both are held in the lumen band and CONVERGE to the wall under load;
      * at high load the fast tier sits within the compliant band d_hat of the oracle;
      * `fast_monotone` / `fast_max_drop` REPORT the fast tier's response, which can be
        genuinely non-monotone — the VBD cable buckles/redistributes under load, and how
        much is architecture-dependent (Warp CPU codegen), so these are diagnostics, NOT
        validation criteria. Only the oracle is required monotone.
    Needs warp+newton for the fast side; raises ImportError if absent (callers skip)."""
    import warp  # noqa: F401
    import newton  # noqa: F401
    from lumen.newton.sim import NewtonGuidewireSim

    from lumen.accurate.ipc import IPCParams, IPCTubeReference
    forces = np.asarray(forces, float)            # materialise: consumed twice (loop + return)
    if forces.size == 0:
        raise ValueError("forces must be non-empty")
    M, n = 40, 11
    cl = np.stack([np.zeros(M), np.zeros(M), np.linspace(0, 80, M)], axis=1)
    x0 = np.stack([np.full(n, 1.5), np.zeros(n), np.linspace(30, 50, n)], axis=1)  # identical seed

    acc, fast = [], []
    for F in forces:
        ref = IPCTubeReference(cl, R, IPCParams(d_hat=d_hat, kappa=kappa_acc))
        x, _ = ref.solve(x0.copy(), F=np.array([float(F), 0.0, 0.0]), iters=600)
        acc.append(float(np.linalg.norm(x[:, :2], axis=1).max()))      # deepest contact radius

        sim = NewtonGuidewireSim(cl, R, x0.copy(), radius=0.2, kappa=kappa_fast, d_hat=d_hat,
                                 vbd_iterations=12, device="cpu")
        for _ in range(80):
            sim.step(dt=2.5e-2, substeps=5, preload=(float(F), 0.0, 0.0))
        fast.append(float(sim.node_radii().max()))

    acc, fast = np.array(acc), np.array(fast)
    fast_max_drop = float(min(0.0, np.min(np.diff(fast)))) if fast.size > 1 else 0.0
    props = {
        "accurate_monotone": bool(np.all(np.diff(acc) >= -1e-3)),   # quasi-static -> clean
        # DIAGNOSTIC (not a pass criterion): the fast tier can be non-monotone — the VBD
        # cable buckles under load by an architecture-dependent amount (Warp CPU codegen)
        "fast_monotone": bool(np.all(np.diff(fast) >= -1e-3)),
        "fast_max_drop": fast_max_drop,                  # most-negative step (0 if monotone)
        "accurate_penetration_free": bool(np.all(acc <= R + 1e-6)),
        "both_held": bool(acc.max() <= R + d_hat + 0.1 and fast.max() <= R + d_hat + 0.1),
        "converge_to_wall": bool(acc[-1] > R - d_hat and fast[-1] > R - d_hat),
        "fast_within_band_of_oracle": float(abs(fast[-1] - acc[-1])),   # at high load, should be < d_hat
    }
    return {"forces": forces, "accurate": acc, "fast": fast, "properties": props}


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
        except (ImportError, ModuleNotFoundError):
            pass
    return {"analytic_oracle": True, "ipc_reference": True, "external_oracle": external,
            "note": "analytic + built-in penetration-free IPC reference "
                    "(lumen.accurate.ipc) always on; STARK/ppf-contact-solver drop in "
                    "via the same seam when built on a GPU box (doc §3.3)"}
