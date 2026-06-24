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


@wp.kernel
def accumulate_tree_barrier(
    color_group: wp.array(dtype=wp.int32),
    wire_mask: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    P: wp.array(dtype=wp.vec3),               # all edges' centerline vertices, concatenated
    Tg: wp.array(dtype=wp.vec3),
    M1: wp.array(dtype=wp.vec3),
    cum_s: wp.array(dtype=wp.float32),        # per-edge cumulative arc-length (each edge starts at 0)
    edge_vstart: wp.array(dtype=wp.int32),    # [n_edges] first vertex index of each edge in P
    edge_vcount: wp.array(dtype=wp.int32),    # [n_edges] vertex count per edge
    edge_smax: wp.array(dtype=wp.float32),    # [n_edges] arc length per edge
    edge_start_junc: wp.array(dtype=wp.int32),  # [n_edges] 1 if the edge's start node is a junction
    edge_end_junc: wp.array(dtype=wp.int32),    # [n_edges] 1 if the edge's end node is a junction
    n_edges: int,
    R0_grid: wp.array(dtype=wp.float32),      # [n_edges*n_s*n_th] branch-BLENDED radius (rigid)
    n_s: int, n_th: int,
    kappa: float, d_hat: float, mode: int,
    mu_along: float, mu_across: float, gamma_fric: float, dt: float,
    body_forces: wp.array(dtype=wp.vec3),
    body_hessian_ll: wp.array(dtype=wp.mat33),
    wall_load: wp.array(dtype=wp.float32),    # [n_edges*n_s*n_th] (out)
):
    """Tree variant of the tube barrier: each wire node finds its nearest segment
    ACROSS ALL EDGES, then reads that edge's blended rigid radius block. The §3.5.2
    branch blending is pre-baked into R0_grid at build time, so the kernel stays
    simple. Junction ends are NOT culled (a node there is transitioning between edges),
    unlike open vessel ends. Single-env, rigid wall (deformable tree wall is future)."""
    t = wp.tid()
    bid = color_group[t]
    if wire_mask[bid] == 0:
        return
    p = wp.transform_get_translation(body_q[bid])
    best = float(1.0e30)
    bj = int(0)
    bu = float(0.0)
    be = int(0)
    # O(n_edges × verts) nearest-segment scan, no spatial cull — fine for procedural
    # small trees; add a per-edge bbox skip before use on full anatomical meshes.
    for e in range(n_edges):
        v0 = edge_vstart[e]
        nv = edge_vcount[e]
        for k in range(nv - 1):
            j = v0 + k
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
                be = e
    v0 = edge_vstart[be]
    vend = v0 + edge_vcount[be] - 1
    # open vessel end -> no contact; but a JUNCTION end is not open (node transits edges)
    if edge_start_junc[be] == 0 and wp.dot(p - P[v0], Tg[v0]) < 0.0:
        return
    if edge_end_junc[be] == 0 and wp.dot(p - P[vend], Tg[vend]) > 0.0:
        return
    a = P[bj]
    foot = a + bu * (P[bj + 1] - a)
    tang = wp.normalize(Tg[bj] + bu * (Tg[bj + 1] - Tg[bj]))
    radial = (p - foot) - wp.dot(p - foot, tang) * tang
    r = wp.length(radial)
    er = radial / (r + 1.0e-9)
    s = cum_s[bj] + bu * (cum_s[bj + 1] - cum_s[bj])
    i_s = wp.clamp(int(s / edge_smax[be] * float(n_s - 1) + 0.5), 0, n_s - 1)
    m1 = M1[bj] - wp.dot(M1[bj], tang) * tang
    m1 = m1 / (wp.length(m1) + 1.0e-9)
    m2 = wp.cross(tang, m1)
    theta = wp.atan2(wp.dot(radial, m2), wp.dot(radial, m1))
    th01 = (theta + wp.pi) / (2.0 * wp.pi)
    i_th = int(th01 * float(n_th)) % n_th
    cell = be * (n_s * n_th) + i_s * n_th + i_th
    R_eff = R0_grid[cell]                      # rigid: blended R baked in (w field is future)
    dwall = R_eff - r
    if dwall < d_hat:
        bp, bpp = _barrier_dd(dwall, d_hat, kappa, mode)
        body_forces[bid] = body_forces[bid] + bp * er
        body_hessian_ll[bid] = body_hessian_ll[bid] + bpp * wp.outer(er, er)
        fn = -bp
        wp.atomic_add(wall_load, cell, fn)
        if mu_across > 0.0 or mu_along > 0.0:
            v_lin = wp.spatial_bottom(body_qd[bid])
            v_t = v_lin - wp.dot(v_lin, er) * er
            vt_mag = wp.length(v_t)
            circ = wp.normalize(wp.cross(tang, er))
            fiber = wp.cos(gamma_fric) * circ + wp.sin(gamma_fric) * tang
            tdir = v_t / vt_mag if vt_mag > 1.0e-9 else circ
            ca = wp.dot(tdir, fiber)
            mu = mu_across + (mu_along - mu_across) * ca * ca
            eps_v = 1.0e-2
            denom = wp.sqrt(vt_mag * vt_mag + eps_v * eps_v)
            body_forces[bid] = body_forces[bid] - (mu * fn / denom) * v_t
            c_t = mu * fn / (denom * dt)
            ident = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            body_hessian_ll[bid] = body_hessian_ll[bid] + c_t * (ident - wp.outer(er, er))
