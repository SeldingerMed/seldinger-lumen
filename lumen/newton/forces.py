"""Small external-force helper for the Newton sim.

`add_world_force` applies a constant world-frame force to a set of bodies (linear
component of the `body_f` wrench, which Newton stores as ``(torque, force)``).
Used by `NewtonGuidewireSim.step(preload=...)` as a test/validation driver that
presses the device into the wall; not part of the contact model itself (the
tube-intrinsic contact lives in `tube_barrier_kernel` + `vbd_fork`).
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def add_world_force(body_ids: wp.array(dtype=wp.int32), fx: float, fy: float, fz: float,
                    skip_first: int, body_f: wp.array(dtype=wp.spatial_vector)):
    k = wp.tid()
    if k < skip_first:
        return
    wp.atomic_add(body_f, body_ids[k],
                  wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), wp.vec3(fx, fy, fz)))


@wp.kernel
def add_body_forces(body_ids: wp.array(dtype=wp.int32),
                    fvecs: wp.array(dtype=wp.vec3), skip_first: int,
                    body_f: wp.array(dtype=wp.spatial_vector)):
    """Add a per-body world-frame linear force (e.g. flow drag along each tangent)."""
    k = wp.tid()
    if k < skip_first:
        return
    wp.atomic_add(body_f, body_ids[k], wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), fvecs[k]))


@wp.func
def _shortest_arc(a: wp.vec3, b: wp.vec3):
    """Unit quaternion rotating unit vector a onto unit vector b (shortest arc).
    Preserves roll about the axis, so applying it to a frame tracks the tangent
    without altering the accumulated twist."""
    d = wp.clamp(wp.dot(a, b), -1.0, 1.0)
    if d > 0.99999:
        return wp.quat_identity()
    if d < -0.99999:                                # antiparallel: any perpendicular axis
        ax = wp.cross(a, wp.vec3(1.0, 0.0, 0.0))
        if wp.length(ax) < 1.0e-6:
            ax = wp.cross(a, wp.vec3(0.0, 1.0, 0.0))
        return wp.quat_from_axis_angle(wp.normalize(ax), 3.14159265)
    return wp.quat_from_axis_angle(wp.normalize(wp.cross(a, b)), wp.acos(d))


@wp.kernel
def actuate_bases(base_ids: wp.array(dtype=wp.int32),
                  insertion: wp.array(dtype=wp.float32),
                  twist: wp.array(dtype=wp.float32),
                  P: wp.array(dtype=wp.vec3),               # centerline vertices
                  Tg: wp.array(dtype=wp.vec3),              # centerline tangents
                  cum_s: wp.array(dtype=wp.float32),        # cumulative arc-length
                  M: int, s_max: float,
                  body_q: wp.array(dtype=wp.transform)):
    """Proximal-end actuation along the CENTERLINE arc-length (one thread per env's
    kinematic base). Insertion advances the base's arc-length s by `insertion`,
    re-seating it on the centerline at the new s while preserving its radial lumen
    offset and tracking the local tangent — so the kinematic base stays inside the
    lumen through curves (translating straight along its own axis drifts it out the
    outer wall of a bend, which contact can't fix because the base is kinematic).
    Twist spins about the new tangent; accumulated twist is preserved (shortest-arc
    tangent tracking doesn't touch roll). The action space of doc §1.2.
    """
    e = wp.tid()
    b = base_ids[e]
    t = body_q[b]
    p = wp.transform_get_translation(t)
    q = wp.transform_get_rotation(t)
    # nearest centerline segment -> current arc-length s and the perpendicular offset
    best = float(1.0e30)
    bj = int(0)
    bu = float(0.0)
    for j in range(M - 1):
        a = P[j]
        ab = P[j + 1] - a
        # guard duplicate/zero-length centerline segments (ab·ab == 0)
        u = wp.clamp(wp.dot(p - a, ab) / wp.max(wp.dot(ab, ab), 1.0e-9), 0.0, 1.0)
        dd = p - (a + u * ab)
        d2 = wp.dot(dd, dd)
        if d2 < best:
            best = d2
            bj = j
            bu = u
    foot = P[bj] + bu * (P[bj + 1] - P[bj])
    tang_old = wp.normalize(Tg[bj] + bu * (Tg[bj + 1] - Tg[bj]))
    offset = p - foot                              # perpendicular to the centerline
    s = cum_s[bj] + bu * (cum_s[bj + 1] - cum_s[bj])
    new_s = wp.clamp(s + insertion[e], 0.0, s_max)
    # segment containing new_s
    k = int(0)
    for j in range(M - 1):
        if cum_s[j] <= new_s:
            k = j
    seg = cum_s[k + 1] - cum_s[k]
    v = wp.clamp((new_s - cum_s[k]) / wp.max(seg, 1.0e-9), 0.0, 1.0)
    foot_new = P[k] + v * (P[k + 1] - P[k])
    tang_new = wp.normalize(Tg[k] + v * (Tg[k + 1] - Tg[k]))
    rot = _shortest_arc(tang_old, tang_new)        # rotate frame + offset to the new tangent
    new_p = foot_new + wp.quat_rotate(rot, offset)
    qn = wp.mul(rot, q)
    tw = twist[e]
    if tw != 0.0:
        qn = wp.mul(wp.quat_from_axis_angle(tang_new, tw), qn)
    body_q[b] = wp.transform(new_p, wp.normalize(qn))
