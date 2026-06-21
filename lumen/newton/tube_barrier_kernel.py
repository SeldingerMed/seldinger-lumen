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
"""

from __future__ import annotations

import warp as wp


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
    P: wp.array(dtype=wp.vec3),
    Tg: wp.array(dtype=wp.vec3),
    M: int,
    R0: float, s_max: float, n_s: int, n_th: int,
    w_field: wp.array(dtype=wp.float32),      # [n_s*n_th] radial displacement (shared R)
    kappa: float, d_hat: float, mode: int,
    mu_along: float, mu_across: float, gamma_fric: float,  # anisotropic friction
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
    a = P[bj]
    foot = a + bu * (P[bj + 1] - a)
    tang = wp.normalize(Tg[bj] + bu * (Tg[bj + 1] - Tg[bj]))
    radial = (p - foot) - wp.dot(p - foot, tang) * tang
    r = wp.length(radial)
    er = radial / (r + 1.0e-9)

    # arc-length s of this contact (segment cum-length approximated by index*seg)
    s = (float(bj) + bu) * (s_max / float(M - 1))
    i_s = wp.clamp(int(s / s_max * float(n_s - 1) + 0.5), 0, n_s - 1)
    m1 = wp.vec3(1.0, 0.0, 0.0)               # reference axis for theta binning
    theta = wp.atan2(wp.dot(radial, wp.cross(tang, m1)), wp.dot(radial, m1))
    th01 = (theta + 3.14159265) / 6.2831853
    i_th = int(th01 * float(n_th)) % n_th
    cell = i_s * n_th + i_th

    R_eff = R0 + w_field[cell]                 # SHARED deformable radius
    dwall = R_eff - r
    if dwall < d_hat:
        bp, bpp = _barrier_dd(dwall, d_hat, kappa, mode)
        body_forces[bid] = body_forces[bid] + bp * er
        body_hessian_ll[bid] = body_hessian_ll[bid] + bpp * wp.outer(er, er)
        fn = -bp                               # normal load magnitude (>0)
        wp.atomic_add(wall_load, cell, fn)
        # --- anisotropic, fiber-aligned friction (doc §3.5.5) ---
        if mu_across > 0.0 or mu_along > 0.0:
            v_lin = wp.spatial_bottom(body_qd[bid])         # device node linear velocity
            v_t = v_lin - wp.dot(v_lin, er) * er            # wall-tangential component
            vt_mag = wp.length(v_t)
            if vt_mag > 1.0e-6:
                tdir = v_t / vt_mag
                circ = wp.normalize(wp.cross(tang, er))     # circumferential direction
                fiber = wp.cos(gamma_fric) * circ + wp.sin(gamma_fric) * tang  # HGO fiber
                ca = wp.dot(tdir, fiber)
                mu = mu_across + (mu_along - mu_across) * ca * ca  # min along fibers
                # regularised Coulomb force opposing tangential sliding
                body_forces[bid] = body_forces[bid] - (mu * fn) * tdir
