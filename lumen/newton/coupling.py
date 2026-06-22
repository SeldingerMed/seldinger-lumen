"""On-device per-substep coupling kernels (M3): keep the wall/clot/flow co-sim on
the GPU so batched envs don't pay a host round-trip every substep.

- compose_radius_k: r0(s,θ) = R0_base·pulse − clot_occlusion, and the clot mask
  (cells the clot bears, so the HGO wall skips them — H1), written straight into the
  wall's device fields.
- flow_drag_k: apply each device node's LOCAL axial drag (∝ interpolated flow
  velocity v(s)) as a body force, reading the flow's device velocity field.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def compose_radius_k(
    R0_base: wp.array(dtype=wp.float32),       # [n_envs*n_s*n_th] resting radius
    occ: wp.array(dtype=wp.float32),           # [n_envs*n_s] clot occlusion o(s) (0 if none)
    pulse: float, n_s: int, n_th: int,
    r0_out: wp.array(dtype=wp.float32),        # [n_envs*n_s*n_th] -> contact kernel reads this
    clot_mask_out: wp.array(dtype=wp.float32)):
    c = wp.tid()
    ncell = n_s * n_th
    env = c // ncell
    i_s = (c % ncell) // n_th
    o = occ[env * n_s + i_s]
    r0_out[c] = R0_base[c] * pulse - o
    if o > 1.0e-9:
        clot_mask_out[c] = 0.0                  # clot cell: wall skips this load (H1)
    else:
        clot_mask_out[c] = 1.0


@wp.kernel
def flow_drag_k(
    s_nodes: wp.array(dtype=wp.float32),       # [n_envs*n_per_env] node arc-length
    tang: wp.array(dtype=wp.vec3),             # [n_envs*n_per_env] node tangent (world)
    body_ids: wp.array(dtype=wp.int32),        # [n_envs*n_per_env] -> body index
    v: wp.array(dtype=wp.float32),             # [n_envs*n_s] flow velocity field
    n_per_env: int, n_s: int, s_max: float, drag_coeff: float,
    body_f: wp.array(dtype=wp.spatial_vector)):
    k = wp.tid()
    env = k // n_per_env
    x = wp.clamp(s_nodes[k] / s_max * float(n_s - 1), 0.0, float(n_s - 1))
    i0 = int(x)
    i1 = wp.min(i0 + 1, n_s - 1)
    frac = x - float(i0)
    vq = v[env * n_s + i0] * (1.0 - frac) + v[env * n_s + i1] * frac
    f = drag_coeff * vq
    wp.atomic_add(body_f, body_ids[k],
                  wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f * tang[k]))
