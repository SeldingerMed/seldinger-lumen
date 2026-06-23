"""Digitally reconstructed radiograph: ray-march a μ volume from a C-arm (Layer 1).

For each detector pixel, the ray from the source is clipped to the volume box, sampled
at N points, μ trilinearly interpolated and integrated → the DRR line integral
A = ∫ μ dl. The displayed radiograph is Beer–Lambert I = I₀·exp(−A) (dense device /
contrast → dark). This is the DiffDRR-style path the doc starts from (§4.1); it is a
plain attenuation ray-trace, deliberately low-integration-risk, and ports to a Warp
autodiff kernel for the differentiable registration / device-as-sensor loops (L1.1+).
"""

from __future__ import annotations

import numpy as np

from lumen.sensors.volume import Grid


def _trilinear(vals, lo, spacing, pts):
    """Sample grid `vals` (nx,ny,nz) at world `pts` (...,3); 0 outside. Vectorized."""
    g = (pts - lo) / spacing                               # fractional grid coords
    nx, ny, nz = vals.shape
    i0 = np.floor(g).astype(int)
    fr = g - i0
    out = np.zeros(pts.shape[:-1], float)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = i0[..., 0] + dx; iy = i0[..., 1] + dy; iz = i0[..., 2] + dz
                inb = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz))
                w = (np.where(dx, fr[..., 0], 1 - fr[..., 0])
                     * np.where(dy, fr[..., 1], 1 - fr[..., 1])
                     * np.where(dz, fr[..., 2], 1 - fr[..., 2]))
                ixc = np.clip(ix, 0, nx - 1); iyc = np.clip(iy, 0, ny - 1); izc = np.clip(iz, 0, nz - 1)
                out += np.where(inb, w * vals[ixc, iyc, izc], 0.0)
    return out


def _ray_box(origin, dirs, lo, hi):
    """Slab ray-box intersection. dirs (...,3) -> (tnear, tfar, hit).

    Textbook robust: a ray parallel to a slab (dir component 0) gets a ±inf interval
    for that axis (1/0 -> ±inf), so the axis only constrains when the origin is
    outside the slab. tnear/tfar are real distances (no 1e12 sentinels) so L1.1 can
    safely reuse them for step-size / early-exit."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs                                   # parallel axis -> ±inf
        t0 = (lo - origin) * inv
        t1 = (hi - origin) * inv
    lo_t = np.minimum(t0, t1)
    hi_t = np.maximum(t0, t1)
    # 0*inf -> nan only when the ray lies exactly on a face plane; non-constraining
    lo_t = np.where(np.isnan(lo_t), -np.inf, lo_t)
    hi_t = np.where(np.isnan(hi_t), np.inf, hi_t)
    tmin = np.maximum.reduce(lo_t, axis=-1)
    tmax = np.minimum.reduce(hi_t, axis=-1)
    tnear = np.maximum(tmin, 0.0)
    return tnear, tmax, (tmax > tnear) & np.isfinite(tnear)


def raycast(mu, grid: Grid, carm, n_samples=192):
    """Render the DRR line-integral image A (nv, nu) for a μ volume and C-arm."""
    lo, hi, sp = np.asarray(grid.lo), np.asarray(grid.hi), grid.spacing
    src, dirs = carm.rays()                                # src (3,), dirs (nv,nu,3)
    tnear, tfar, hit = _ray_box(src, dirs, lo, hi)         # (nv,nu)
    ts = np.linspace(0.0, 1.0, n_samples)                  # param along [tnear,tfar]
    seg = (tfar - tnear)                                   # (nv,nu)
    dl = np.where(hit, seg / n_samples, 0.0)
    A = np.zeros(dirs.shape[:2], float)
    for t in ts:
        tt = tnear + t * seg                               # (nv,nu)
        pts = src[None, None, :] + tt[..., None] * dirs    # (nv,nu,3)
        A += _trilinear(mu, lo, sp, pts) * dl
    return A * hit


def radiograph(A):
    """Beer–Lambert intensity from the DRR line integral (I0=1; dense → dark). A
    calibrated I0/detector-response knob lands when sim-to-real calibration needs it."""
    return np.exp(-np.asarray(A))
