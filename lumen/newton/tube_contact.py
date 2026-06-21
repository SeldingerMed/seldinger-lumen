"""Tube-intrinsic contact as a Newton body-force kernel (doc §3.2, §3.5).

The bible's build decision (§3.2): do not write an engine; author the
domain-specialized contact as a module *inside* Newton. Here that is a Warp
kernel that, each substep, reads the guidewire bodies' transforms
(``state.body_q``), projects each into the vessel's tube-intrinsic frame
``(s, θ, r)``, evaluates the analytic barrier, and accumulates the contact
reaction into ``state.body_f`` (the world-frame wrench Newton's SolverVBD then
integrates). This replaces generic device-vs-mesh collision with the
tube-intrinsic narrowphase.

Wrench convention (verified against Newton source): ``body_f`` is
``wp.spatial_vector(torque, force)`` — the linear force is the LAST three
components. ``body_qd`` is ``(angular, linear)`` likewise.
"""

from __future__ import annotations

import numpy as np

import warp as wp

wp.init()


@wp.func
def _nearest_on_centerline(p: wp.vec3, P: wp.array(dtype=wp.vec3),
                           Tg: wp.array(dtype=wp.vec3), M: int):
    """Return (foot, tangent, radial, r) for the nearest centerline segment."""
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
    t = wp.normalize(Tg[bj] + bu * (Tg[bj + 1] - Tg[bj]))
    radial = (p - foot) - wp.dot(p - foot, t) * t
    r = wp.length(radial)
    return foot, t, radial, r


@wp.kernel
def tube_contact_body_force(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_ids: wp.array(dtype=wp.int32),     # guidewire body indices
    P: wp.array(dtype=wp.vec3),             # vessel centerline vertices
    Tg: wp.array(dtype=wp.vec3),            # vessel centerline tangents
    M: int, R: float, kappa: float, d_hat: float, c_damp: float,
    body_f: wp.array(dtype=wp.spatial_vector),
    gap_out: wp.array(dtype=float),         # per-guidewire-node gap (diagnostic)
):
    k = wp.tid()
    bid = body_ids[k]
    p = wp.transform_get_translation(body_q[bid])
    foot, t, radial, r = _nearest_on_centerline(p, P, Tg, M)
    er = radial / (r + 1.0e-9)
    g = R - r                                # gap; <0 = outside lumen wall
    delta = wp.max(d_hat - g, 0.0)           # penetration into the barrier band
    if delta > 0.0:
        v_lin = wp.spatial_bottom(body_qd[bid])   # linear velocity (last 3)
        vr = wp.dot(v_lin, er)               # outward radial velocity (>0 = leaving)
        # bounded compliant barrier + normal damping (dissipates contact energy
        # so the explicit force is stable in VBD's predictor; doc §3.5.3 fast tier)
        fn = wp.max(kappa * delta + c_damp * vr, 0.0)
        f = -fn * er                         # reaction inward (toward axis)
        wp.atomic_add(body_f, bid, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f))
    gap_out[k] = g


@wp.kernel
def add_world_force(body_ids: wp.array(dtype=wp.int32), fx: float, fy: float, fz: float,
                    skip_first: int, body_f: wp.array(dtype=wp.spatial_vector)):
    k = wp.tid()
    if k < skip_first:
        return
    wp.atomic_add(body_f, body_ids[k],
                  wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), wp.vec3(fx, fy, fz)))


class TubeContact:
    """Holds the vessel frame as Warp arrays and launches the contact kernel."""

    def __init__(self, centerline: np.ndarray, R: float, device: str):
        from lumen.core.frame import CenterlineFrame
        f = CenterlineFrame(centerline)
        self.P = wp.array(f.points.astype(np.float32), dtype=wp.vec3, device=device)
        self.Tg = wp.array(f.tangents.astype(np.float32), dtype=wp.vec3, device=device)
        self.M = len(f.points)
        self.R = float(R)
        self.device = device

    def apply(self, state, body_ids: wp.array, gap_out: wp.array,
              kappa: float, d_hat: float, c_damp: float = 0.0):
        wp.launch(tube_contact_body_force, dim=body_ids.shape[0],
                  inputs=[state.body_q, state.body_qd, body_ids, self.P, self.Tg,
                          self.M, self.R, kappa, d_hat, c_damp],
                  outputs=[state.body_f, gap_out], device=self.device)
