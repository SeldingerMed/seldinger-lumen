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
