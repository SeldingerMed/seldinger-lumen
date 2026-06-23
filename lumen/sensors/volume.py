"""Scene → attenuation (μ) voxel grid for the DRR raycast (Layer 1).

The DiffDRR-style path renders by ray-marching a μ volume. We rasterise the parametric
scene into that volume: the radio-opaque device (a capsule chain along its node
polyline) gets high μ; the contrast-filled lumen (a tube over the shared R(s,θ)) gets
contrast μ; tissue/air is ~0 (a clean DSA-like scene). The device uses a SMOOTH
capsule indicator (tanh falloff): a continuous μ field with no hard edge, structured
so the autodiff port (L1.1, Warp) can take gradients w.r.t. the node positions for
the device-as-sensor loop (doc §3.6). Today this is plain numpy — smooth, not yet an
autograd graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    lo: np.ndarray            # (3,) lower corner
    hi: np.ndarray            # (3,) upper corner
    res: tuple                # (nx, ny, nz)

    def centers(self):
        nx, ny, nz = self.res
        xs = np.linspace(self.lo[0], self.hi[0], nx)
        ys = np.linspace(self.lo[1], self.hi[1], ny)
        zs = np.linspace(self.lo[2], self.hi[2], nz)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([gx, gy, gz], axis=-1)             # (nx,ny,nz,3)

    @property
    def spacing(self):                                      # guard res==1 (no /0 warning)
        return (self.hi - self.lo) / np.maximum(np.array(self.res) - 1, 1)


def grid_for(points, margin=8.0, res=64):
    """An axis-aligned grid bounding `points` with a margin; res scalar or (nx,ny,nz)."""
    p = np.asarray(points, float)
    lo, hi = p.min(0) - margin, p.max(0) + margin
    r = (res, res, res) if np.isscalar(res) else tuple(res)
    if any(c <= 1 for c in r):
        raise ValueError(f"all resolution components must be > 1, got {r}")
    return Grid(lo=lo, hi=hi, res=r)


def _point_segment_dist2(P, a, b):
    """Squared distance from points P (...,3) to segment [a,b]. Vectorized."""
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab + 1e-12), 0.0, 1.0)
    foot = a + t[..., None] * ab
    d = P - foot
    return np.einsum("...j,...j->...", d, d)


def voxelize_device(nodes, radius, grid: Grid, mu_device=1.0, eps=0.6):
    """Soft capsule-chain μ volume for the device. Returns (nx,ny,nz) float array.

    μ = mu_device · ½(1 + tanh((radius − d)/eps)) where d is the distance to the
    device polyline — a smooth ~indicator of "inside the wire" (the eps band is the
    soft edge). A single-node device is rendered as a sphere (distance to the point)
    so a degenerate tip-only state still renders rather than silently blanking."""
    nodes = np.asarray(nodes, float)
    if len(nodes) == 0:
        raise ValueError("device must have at least one node")
    C = grid.centers()                                     # (nx,ny,nz,3)
    flat = C.reshape(-1, 3)
    if len(nodes) == 1:                                    # H1: sphere about the lone node
        diff = flat - nodes[0]
        d2 = np.einsum("ij,ij->i", diff, diff)
    else:
        d2 = np.full(len(flat), np.inf)
        for a, b in zip(nodes[:-1], nodes[1:]):
            d2 = np.minimum(d2, _point_segment_dist2(flat, a, b))
    d = np.sqrt(d2)
    mu = mu_device * 0.5 * (1.0 + np.tanh((radius - d) / eps))
    return mu.reshape(grid.res)
