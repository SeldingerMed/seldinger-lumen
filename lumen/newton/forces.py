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


@wp.kernel
def actuate_bases(base_ids: wp.array(dtype=wp.int32),
                  insertion: wp.array(dtype=wp.float32),
                  twist: wp.array(dtype=wp.float32),
                  body_q: wp.array(dtype=wp.transform)):
    """Proximal-end actuation, on device (one thread per env's kinematic base).

    Insertion translates the base along its CURRENT axis (the capsule local +z
    rotated into world — correct on curves); twist spins it about that axis. This
    is the device's action space (doc §1.2). Done in a kernel so it adds no per-step
    host round-trip and batches over envs.
    """
    e = wp.tid()
    b = base_ids[e]
    t = body_q[b]
    p = wp.transform_get_translation(t)
    q = wp.transform_get_rotation(t)
    axis = wp.normalize(wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)))
    p = p + insertion[e] * axis
    tw = twist[e]
    if tw != 0.0:
        q = wp.normalize(wp.mul(wp.quat_from_axis_angle(axis, tw), q))
    body_q[b] = wp.transform(p, q)
