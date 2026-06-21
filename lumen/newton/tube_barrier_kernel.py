"""Tube-intrinsic barrier as a native AVBD constraint (force + Hessian).

This kernel is injected into the forked SolverVBD's per-color rigid solve. Unlike
an external predictor force, it adds BOTH the barrier reaction force and its
Hessian (κ·eᵣ⊗eᵣ) to the per-body 6×6 system, so contact is treated *implicitly*
together with inertia and the cable joints — which is what makes stiff contact
stable in VBD (doc §3.5: barrier inside the solve, not as an explicit force).

Energy (compliant fast tier, doc §3.5.3): E = ½·κ·δ², δ = max(0, d_hat − (R − r)),
where r is the tube-intrinsic radius. Force = −∂E/∂p = −κ·δ·eᵣ (inward);
Hessian ≈ κ·eᵣ⊗eᵣ (SPD, preserves the system's positive-definiteness).
"""

from __future__ import annotations

import warp as wp


@wp.func
def _barrier_dd(d: float, d_hat: float, kappa: float, mode: int):
    """Return (b'(d), b''(d)) for the wall-distance barrier.

    mode 0 — compliant fast tier: b = ½·κ·(d_hat−d)²  (penetrating, bounded).
             b' = −κ·(d_hat−d),  b'' = κ.
    mode 1 — IPC log barrier form (doc §3.5.3, Li et al. 2020):
             b = −κ·(d−d_hat)²·ln(d/d_hat),  → +∞ as d→0.
             NOTE: rigorous penetration-free IPC requires a CCD filter line search
             and is provided by the ACCURATE TIER (STARK/ppf, doc §3.3 "borrow, do
             not build"). This in-VBD form is an experimental option, bounded here
             so it is numerically safe (it is not the default fast-tier contact).
    Force on the body is b'(d)·eᵣ (caller); b''(d) feeds the SPD Hessian.
    """
    if mode == 1:
        dd = wp.max(d, 0.05 * d_hat)         # floor (no CCD line search in VBD)
        ln = wp.log(dd / d_hat)
        diff = dd - d_hat
        bp = -kappa * (2.0 * diff * ln + diff * diff / dd)
        bpp = -kappa * (2.0 * ln + 4.0 * diff / dd - diff * diff / (dd * dd))
        # bound so a no-line-search step cannot explode (safety, not IPC-rigorous)
        bp = wp.max(bp, -50.0 * kappa * d_hat)
        return bp, wp.clamp(bpp, 0.0, 200.0 * kappa)
    # compliant (fast tier) — the validated default
    return -kappa * (d_hat - d), kappa


@wp.kernel
def accumulate_tube_barrier(
    color_group: wp.array(dtype=wp.int32),
    wire_mask: wp.array(dtype=wp.int32),     # 1 for guidewire bodies, else 0
    body_q: wp.array(dtype=wp.transform),
    P: wp.array(dtype=wp.vec3),              # vessel centerline vertices
    Tg: wp.array(dtype=wp.vec3),             # vessel centerline tangents
    M: int, R: float, kappa: float, d_hat: float, mode: int,
    body_forces: wp.array(dtype=wp.vec3),    # in/out (accumulated)
    body_hessian_ll: wp.array(dtype=wp.mat33),  # in/out (accumulated)
):
    t = wp.tid()
    bid = color_group[t]
    if wire_mask[bid] == 0:
        return
    p = wp.transform_get_translation(body_q[bid])
    # nearest centerline segment (tube-intrinsic narrowphase)
    best = float(1.0e30)
    bj = int(0)
    bu = float(0.0)
    for j in range(M - 1):
        a = P[j]
        ab = P[j + 1] - a
        L2 = wp.dot(ab, ab)
        u = wp.clamp(wp.dot(p - a, ab) / L2, 0.0, 1.0)
        d = p - (a + u * ab)
        d2 = wp.dot(d, d)
        if d2 < best:
            best = d2
            bj = j
            bu = u
    a = P[bj]
    foot = a + bu * (P[bj + 1] - a)
    tang = wp.normalize(Tg[bj] + bu * (Tg[bj + 1] - Tg[bj]))
    radial = (p - foot) - wp.dot(p - foot, tang) * tang
    r = wp.length(radial)
    er = radial / (r + 1.0e-9)
    dwall = R - r                            # distance to the lumen wall (>0 inside)
    if dwall < d_hat:                        # within the barrier band
        bp, bpp = _barrier_dd(dwall, d_hat, kappa, mode)
        # force = -dE/dp = b'(d)·eᵣ (inward, since b'(d) < 0); Hessian = b''(d)·eᵣ⊗eᵣ.
        # one body per thread per color -> non-atomic accumulate is safe.
        body_forces[bid] = body_forces[bid] + bp * er
        body_hessian_ll[bid] = body_hessian_ll[bid] + bpp * wp.outer(er, er)
