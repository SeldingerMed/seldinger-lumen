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


def _segment_projection_t(P, a, b):
    """Projection parameter onto [a, b], stable for degenerate/overflowing segments."""
    ab = b - a
    with np.errstate(over="ignore", invalid="ignore"):
        denom = float(ab @ ab)
    if not np.isfinite(denom) or denom <= 1e-24:
        return np.zeros(P.shape[:-1], dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        t = ((P - a) @ ab) / denom
    return np.clip(np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _point_segment_dist2(P, a, b):
    """Squared distance from points P (...,3) to segment [a,b]. Vectorized."""
    ab = b - a
    t = _segment_projection_t(P, a, b)
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


def _segment_dist_and_t(P, a, b):
    ab = b - a
    t = _segment_projection_t(P, a, b)
    foot = a + t[..., None] * ab
    d = P - foot
    return np.einsum("...j,...j->...", d, d), t


def voxelize_polyline(nodes, radii, grid: Grid, mu_device=1.0, eps=0.6):
    """Soft variable-radius capsule-chain μ volume."""
    nodes = np.asarray(nodes, float)
    radii = np.asarray(radii, float)
    if len(nodes) == 0:
        raise ValueError("polyline must have at least one node")
    if len(radii) != len(nodes):
        raise ValueError("radii must have one value per polyline node")
    C = grid.centers()
    flat = C.reshape(-1, 3)
    fill = np.zeros(len(flat), float)
    if len(nodes) == 1:
        diff = flat - nodes[0]
        d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        fill = np.maximum(fill, 0.5 * (1.0 + np.tanh((radii[0] - d) / eps)))
    else:
        for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
            d2, t = _segment_dist_and_t(flat, a, b)
            r = radii[i] * (1.0 - t) + radii[i + 1] * t
            fill = np.maximum(fill, 0.5 * (1.0 + np.tanh((r - np.sqrt(d2)) / eps)))
    return (mu_device * fill).reshape(grid.res)


def edge_radii(edge, pts):
    """Radius at each stored centerline point, interpolated from an asset edge."""
    pts = np.asarray(pts, float)
    if len(pts) <= 1:
        return np.full(len(pts), float(np.asarray(edge.R, float).mean()))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_pts = np.concatenate([[0.0], np.cumsum(seg)])
    s_grid = np.asarray(edge.s_grid, float)
    r_grid = np.asarray(edge.R, float).mean(axis=1)
    return np.interp(s_pts, s_grid, r_grid)


def asset_points(asset):
    """All centerline points in an asset, for C-arm/grid fitting."""
    pts = []
    for edge in asset.edges:
        pts.extend(np.asarray(edge.centerline_mm, float))
    if not pts:
        raise ValueError("asset has no edge centerline points")
    return np.asarray(pts, float)


def voxelize_asset(asset, grid: Grid, mu_device=1.0, eps=0.6):
    """Rasterize every edge in a Lumen asset into a μ volume."""
    mu = np.zeros(grid.res, float)
    for edge in asset.edges:
        pts = np.asarray(edge.centerline_mm, float)
        radii = edge_radii(edge, pts)
        mu = np.maximum(mu, voxelize_polyline(pts, radii, grid,
                                              mu_device=mu_device, eps=eps))
    return mu
