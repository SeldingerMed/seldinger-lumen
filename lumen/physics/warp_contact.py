"""Tube-intrinsic contact as Warp kernels (the doc's intended substrate, §3.5.9).

This is the real fast-tier narrowphase: one GPU thread per device node, projecting
into the tube-intrinsic frame, evaluating the analytic barrier, writing the
per-node contact force -- batched across thousands of environments and
differentiable via Warp's autodiff tape. The same kernel source runs on CUDA
(GPU throughput) and on the Warp CPU device (used here for correctness tests and
as the no-GPU path); the PyTorch implementation in contact.py is the final
fallback when Warp is unavailable.

Mirrors the kernel sketch in doc §3.5.9. R is the (constant, cylinder) lumen
radius for v1; an R(s) interpolation is a small extension that reads a per-segment
radius array.
"""

from __future__ import annotations

import numpy as np

import warp as wp

wp.init()


@wp.kernel
def _contact_force_kernel(
    X: wp.array(dtype=wp.vec3),          # device node positions (flattened B*N)
    P: wp.array(dtype=wp.vec3),          # centerline vertices
    Tg: wp.array(dtype=wp.vec3),         # centerline tangents
    M: int, R: float, kappa: float, d_hat: float,
    force: wp.array(dtype=wp.vec3),      # out: contact force per node
    gap_out: wp.array(dtype=float),      # out: gap per node
):
    i = wp.tid()
    p = X[i]
    best = float(1.0e30)
    bj = int(0)
    bu = float(0.0)
    # narrowphase: nearest centerline segment (the O(M) scan; an arc-length
    # broadphase cull would make this O(1) per node -- doc §3.5.4)
    for j in range(M - 1):
        a = P[j]
        ab = P[j + 1] - a
        L2 = wp.dot(ab, ab)
        u = wp.clamp(wp.dot(p - a, ab) / L2, 0.0, 1.0)
        diff = p - (a + u * ab)
        d2 = wp.dot(diff, diff)
        if d2 < best:
            best = d2
            bj = j
            bu = u
    a = P[bj]
    foot = a + bu * (P[bj + 1] - a)
    t = wp.normalize(Tg[bj] + bu * (Tg[bj + 1] - Tg[bj]))
    radial = (p - foot) - wp.dot(p - foot, t) * t
    r = wp.length(radial)
    er = radial / (r + 1.0e-12)
    g = R - r
    fn = kappa * wp.max(d_hat - g, 0.0)
    force[i] = -fn * er
    gap_out[i] = g


@wp.kernel
def _barrier_energy_kernel(
    X: wp.array(dtype=wp.vec3),
    P: wp.array(dtype=wp.vec3),
    Tg: wp.array(dtype=wp.vec3),
    M: int, R: float, kappa: float, d_hat: float,
    energy: wp.array(dtype=float),       # out: scalar (atomic-summed) for autodiff
):
    i = wp.tid()
    p = X[i]
    best = float(1.0e30)
    bj = int(0)
    bu = float(0.0)
    for j in range(M - 1):
        a = P[j]
        ab = P[j + 1] - a
        L2 = wp.dot(ab, ab)
        u = wp.clamp(wp.dot(p - a, ab) / L2, 0.0, 1.0)
        diff = p - (a + u * ab)
        d2 = wp.dot(diff, diff)
        if d2 < best:
            best = d2
            bj = j
            bu = u
    a = P[bj]
    foot = a + bu * (P[bj + 1] - a)
    t = wp.normalize(Tg[bj] + bu * (Tg[bj + 1] - Tg[bj]))
    radial = (p - foot) - wp.dot(p - foot, t) * t
    r = wp.length(radial)
    pen = wp.max(d_hat - (R - r), 0.0)
    wp.atomic_add(energy, 0, 0.5 * kappa * pen * pen)


class WarpTubeContact:
    """Warp narrowphase over a fixed centerline (cylinder radius R)."""

    def __init__(self, centerline: np.ndarray, R: float, device: str | None = None):
        self.device = device or ("cuda" if wp.get_cuda_device_count() > 0 else "cpu")
        from lumen.core.frame import CenterlineFrame
        f = CenterlineFrame(centerline)
        self.P = wp.array(f.points.astype(np.float32), dtype=wp.vec3, device=self.device)
        self.Tg = wp.array(f.tangents.astype(np.float32), dtype=wp.vec3, device=self.device)
        self.M = len(f.points)
        self.R = float(R)

    def forces(self, x: np.ndarray, kappa=1.5e3, d_hat=0.25):
        """x [B, N, 3] -> (gap [B, N], force [B, N, 3]) via the analytic kernel."""
        B, N, _ = x.shape
        X = wp.array(x.reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=self.device)
        force = wp.zeros(B * N, dtype=wp.vec3, device=self.device)
        gap = wp.zeros(B * N, dtype=float, device=self.device)
        wp.launch(_contact_force_kernel, dim=B * N,
                  inputs=[X, self.P, self.Tg, self.M, self.R, kappa, d_hat],
                  outputs=[force, gap], device=self.device)
        wp.synchronize_device(self.device)
        return (gap.numpy().reshape(B, N),
                force.numpy().reshape(B, N, 3))

    def barrier_energy_and_grad(self, x: np.ndarray, kappa=1.5e3, d_hat=0.25):
        """Total barrier energy and dE/dx via Warp's autodiff tape.

        Returns (energy float, grad [B, N, 3]). -grad is the contact force, which
        must match `forces()` -- the differentiability check.
        """
        B, N, _ = x.shape
        X = wp.array(x.reshape(-1, 3).astype(np.float32), dtype=wp.vec3,
                     device=self.device, requires_grad=True)
        energy = wp.zeros(1, dtype=float, device=self.device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(_barrier_energy_kernel, dim=B * N,
                      inputs=[X, self.P, self.Tg, self.M, self.R, kappa, d_hat],
                      outputs=[energy], device=self.device)
        tape.backward(loss=energy)
        e = float(energy.numpy()[0])
        grad = X.grad.numpy().reshape(B, N, 3)
        return e, grad
