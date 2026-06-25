"""Tube-intrinsic barrier as a native AVBD constraint, over a deformable wall.

Injected into the forked SolverVBD's per-color rigid solve. Adds BOTH the barrier
reaction force and its Hessian (κ·eᵣ⊗eᵣ) to the per-body 6×6 system so contact is
treated *implicitly* with inertia + cable joints — what makes stiff contact stable
in VBD (doc §3.5). The lumen radius is the SHARED field R(s,θ) = R0 + w(s,θ): the
barrier reads the deformed radius and deposits the contact normal load onto the
wall cell, so the HGO wall (lumen.newton.hgo_wall) and the contact share R
(doc §3.5.6). With w≡0 the wall is rigid.

Barrier (doc §3.5.3): compliant fast tier E=½κδ² or bounded IPC-log option; the
rigorous penetration-free IPC is the accurate tier (§3.3).

Precision (#21): geometry arrays are float32 here (Warp GPU throughput), while
lumen.core.frame is float64. The resulting s/r differences are ~1e-6 at the
scales we run — negligible and an intentional speed/precision trade. Promote the
arrays to float64 only if very long centerlines or large coordinates demand it.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def accumulate_coaxial_coupling(
    color_group: wp.array(dtype=wp.int32),
    gw_mask: wp.array(dtype=wp.int32),         # 1 for guidewire bodies (the ones constrained)
    body_q: wp.array(dtype=wp.transform),
    cath_ids: wp.array(dtype=wp.int32),        # catheter body indices, ordered along the rod
    n_cath: int,
    r_inner: float,                            # catheter inner-lumen radius the gw rides within
    kappa: float, d_hat: float,
    body_forces: wp.array(dtype=wp.vec3),
    body_hessian_ll: wp.array(dtype=wp.mat33),
):
    """Sliding coaxial coupling (L0d.2b): keep each guidewire node within the catheter's
    inner lumen. The catheter centerline is read LIVE from body_q (no host rebuild) — so
    as the catheter bends, the gw is barriered to follow, while sliding freely along the
    axis (no tangential force). Structurally the tube barrier, but the 'wall' is the
    dynamic catheter axis and the barrier pulls INWARD (gw stays inside, r < r_inner).

    One-way for now: only the guidewire feels the constraint (the stiffer catheter is the
    support). Two-way reaction onto the catheter is a future refinement (doc §3.5)."""
    t = wp.tid()
    bid = color_group[t]
    if gw_mask[bid] == 0:
        return
    p = wp.transform_get_translation(body_q[bid])
    best = float(1.0e30)
    bk = int(0)
    bu = float(0.0)
    for k in range(n_cath - 1):
        a = wp.transform_get_translation(body_q[cath_ids[k]])
        ab = wp.transform_get_translation(body_q[cath_ids[k + 1]]) - a
        L2 = wp.dot(ab, ab)
        u = wp.clamp(wp.dot(p - a, ab) / (L2 + 1.0e-12), 0.0, 1.0)
        dd = p - (a + u * ab)
        d2 = wp.dot(dd, dd)
        if d2 < best:
            best = d2
            bk = k
            bu = u
    a = wp.transform_get_translation(body_q[cath_ids[bk]])
    b = wp.transform_get_translation(body_q[cath_ids[bk + 1]])
    foot = a + bu * (b - a)
    tang = wp.normalize(b - a)
    radial = (p - foot) - wp.dot(p - foot, tang) * tang
    r = wp.length(radial)
    er = radial / (r + 1.0e-9)
    dwall = r_inner - r                          # clearance to the catheter inner wall
    if dwall < d_hat:
        bp = -kappa * (d_hat - dwall)            # compliant barrier, pulls inward (bp<0, er outward)
        body_forces[bid] = body_forces[bid] + bp * er
        body_hessian_ll[bid] = body_hessian_ll[bid] + kappa * wp.outer(er, er)


@wp.func
def _barrier_dd(d: float, d_hat: float, kappa: float, mode: int):
    """Return (b'(d), b''(d)) for the wall-distance barrier (see module docstring)."""
    if mode == 1:
        dd = wp.max(d, 0.05 * d_hat)
        ln = wp.log(dd / d_hat)
        diff = dd - d_hat
        bp = -kappa * (2.0 * diff * ln + diff * diff / dd)
        bpp = -kappa * (2.0 * ln + 4.0 * diff / dd - diff * diff / (dd * dd))
        bp = wp.max(bp, -50.0 * kappa * d_hat)
        return bp, wp.clamp(bpp, 0.0, 200.0 * kappa)
    return -kappa * (d_hat - d), kappa


@wp.kernel
def accumulate_tube_barrier(
    color_group: wp.array(dtype=wp.int32),
    wire_mask: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    P: wp.array(dtype=wp.vec3),               # centerline vertices
    Tg: wp.array(dtype=wp.vec3),              # centerline tangents
    M1: wp.array(dtype=wp.vec3),              # rotation-minimizing reference normals (per vertex)
    cum_s: wp.array(dtype=wp.float32),        # cumulative arc-length (per vertex)
    M: int,
    R0_grid: wp.array(dtype=wp.float32),      # [n_envs*n_s*n_th] BASE lumen radius R0(s,θ)
    s_max: float, n_s: int, n_th: int, n_per_env: int,
    w_field: wp.array(dtype=wp.float32),      # [n_envs*n_s*n_th] radial displacement (shared R)
    kappa: float, d_hat: float, mode: int,
    mu_along: float, mu_across: float, gamma_fric: float, dt: float,  # anisotropic friction
    body_forces: wp.array(dtype=wp.vec3),     # in/out
    body_hessian_ll: wp.array(dtype=wp.mat33),  # in/out
    wall_load: wp.array(dtype=wp.float32),    # [n_s*n_th] accumulated normal load (out)
):
    t = wp.tid()
    bid = color_group[t]
    if wire_mask[bid] == 0:
        return
    p = wp.transform_get_translation(body_q[bid])
    best = float(1.0e30)
    bj = int(0)
    bu = float(0.0)
    for j in range(M - 1):
        a = P[j]
        ab = P[j + 1] - a
        L2 = wp.dot(ab, ab)
        u = wp.clamp(wp.dot(p - a, ab) / L2, 0.0, 1.0)
        dd = p - (a + u * ab)
        d2 = wp.dot(dd, dd)
        if d2 < best:
            best = d2
            bj = j
            bu = u
    # #22 — open vessel ends: a node axially past either opening has left the
    # vessel; no wall contact there (don't deposit load at a boundary cell).
    if wp.dot(p - P[0], Tg[0]) < 0.0 or wp.dot(p - P[M - 1], Tg[M - 1]) > 0.0:
        return
    a = P[bj]
    foot = a + bu * (P[bj + 1] - a)
    tang = wp.normalize(Tg[bj] + bu * (Tg[bj + 1] - Tg[bj]))
    radial = (p - foot) - wp.dot(p - foot, tang) * tang
    r = wp.length(radial)
    er = radial / (r + 1.0e-9)

    # arc-length from the true cumulative lengths (matches lumen.core.frame)
    s = cum_s[bj] + bu * (cum_s[bj + 1] - cum_s[bj])
    i_s = wp.clamp(int(s / s_max * float(n_s - 1) + 0.5), 0, n_s - 1)
    # theta in the rotation-minimizing frame (non-degenerate for any centerline)
    m1 = M1[bj] - wp.dot(M1[bj], tang) * tang
    m1 = m1 / (wp.length(m1) + 1.0e-9)
    m2 = wp.cross(tang, m1)
    theta = wp.atan2(wp.dot(radial, m2), wp.dot(radial, m1))
    th01 = (theta + 3.14159265) / 6.2831853
    i_th = int(th01 * float(n_th)) % n_th
    # per-env wall block: each env has its own R0/w/load grid of n_s*n_th cells, so a
    # body's contact reads/writes its OWN env's wall (env = body id / bodies-per-env).
    env = bid // n_per_env
    cell = env * (n_s * n_th) + i_s * n_th + i_th

    R_eff = R0_grid[cell] + w_field[cell]      # SHARED radius: base R0(s,θ) + deformation
    dwall = R_eff - r
    if dwall < d_hat:
        bp, bpp = _barrier_dd(dwall, d_hat, kappa, mode)
        body_forces[bid] = body_forces[bid] + bp * er
        body_hessian_ll[bid] = body_hessian_ll[bid] + bpp * wp.outer(er, er)
        fn = -bp                               # normal load magnitude (>0)
        wp.atomic_add(wall_load, cell, fn)
        # --- anisotropic, fiber-aligned friction, implicit (doc §3.5.5) ---
        if mu_across > 0.0 or mu_along > 0.0:
            v_lin = wp.spatial_bottom(body_qd[bid])         # device node linear velocity
            v_t = v_lin - wp.dot(v_lin, er) * er            # wall-tangential component
            vt_mag = wp.length(v_t)
            # fiber-aligned mu (min along fibers); default to a circumferential
            # slide direction when not yet moving, so mu is well-defined.
            circ = wp.normalize(wp.cross(tang, er))
            fiber = wp.cos(gamma_fric) * circ + wp.sin(gamma_fric) * tang
            tdir = v_t / vt_mag if vt_mag > 1.0e-9 else circ
            ca = wp.dot(tdir, fiber)
            mu = mu_across + (mu_along - mu_across) * ca * ca
            # Stribeck-regularised Coulomb: smooth through v_t=0 (sticking band of
            # width eps_v), so force is bounded and the tangent stiffness below is
            # finite -> implicit, stable at high mu (the explicit version was not).
            eps_v = 1.0e-2
            denom = wp.sqrt(vt_mag * vt_mag + eps_v * eps_v)
            body_forces[bid] = body_forces[bid] - (mu * fn / denom) * v_t
            # friction (tangent-space) Hessian, PSD, added so the AVBD solve treats
            # friction implicitly (#19). v_t ≈ Δx/dt, so the position-Hessian of the
            # velocity-based friction force carries a 1/dt factor (L7).
            c_t = mu * fn / (denom * dt)
            ident = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            body_hessian_ll[bid] = body_hessian_ll[bid] + c_t * (ident - wp.outer(er, er))
