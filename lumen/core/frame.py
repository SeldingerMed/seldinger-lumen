"""Tube-intrinsic coordinate frame.

Maps a world-space point to ``(s, theta, r)`` along a centerline:

    s      arc-length from the centerline start
    theta  circumferential angle in a rotation-minimizing frame
    r      radial distance from the centerline

This is the modality-agnostic geometric core (doc Layer 0 §3.5.2). It knows
nothing about vessels, airways, or any specific anatomy -- it operates on a bare
centerline polyline. The lumen boundary R(s, theta) lives separately in
``lumen.core.lumen_field`` so wall mechanics and contact can share one field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _tangents(pts: np.ndarray) -> np.ndarray:
    """Per-vertex unit tangents (central difference, one-sided at the ends)."""
    t = np.zeros_like(pts)
    t[1:-1] = pts[2:] - pts[:-2]
    t[0] = pts[1] - pts[0]
    t[-1] = pts[-1] - pts[-2]
    return np.stack([_unit(ti) for ti in t])


def _rmf(pts: np.ndarray, tang: np.ndarray) -> np.ndarray:
    """Rotation-minimizing frame reference normals (double-reflection, Wang 2008).

    Returns per-vertex unit normals m1 (each orthogonal to its tangent); the
    second reference is m2 = t x m1. Using an RMF instead of a Frenet frame
    avoids the binormal flip at inflection / straight points.
    """
    n = len(pts)
    m1 = np.zeros_like(pts)
    # seed: any unit vector orthogonal to the first tangent
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, tang[0])) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    m1[0] = _unit(seed - np.dot(seed, tang[0]) * tang[0])
    for i in range(n - 1):
        v1 = pts[i + 1] - pts[i]
        c1 = float(np.dot(v1, v1))
        if c1 == 0.0:
            m1[i + 1] = m1[i]
            continue
        r_l = m1[i] - (2.0 / c1) * np.dot(v1, m1[i]) * v1
        t_l = tang[i] - (2.0 / c1) * np.dot(v1, tang[i]) * v1
        v2 = tang[i + 1] - t_l
        c2 = float(np.dot(v2, v2))
        m1[i + 1] = r_l if c2 == 0.0 else r_l - (2.0 / c2) * np.dot(v2, r_l) * v2
        m1[i + 1] = _unit(m1[i + 1])
    return m1


@dataclass
class Projection:
    s: float          # arc-length along the centerline
    theta: float      # circumferential angle, radians in [-pi, pi]
    r: float          # radial distance from the centerline
    e_r: np.ndarray   # unit radial direction in world space
    edge_param: float  # nearest-segment parameter u in [0, 1] (debug/blending)


class CenterlineFrame:
    """Curvilinear frame fitted to one centerline polyline."""

    def __init__(self, points: np.ndarray):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise ValueError("centerline must be an (N>=2, 3) array")
        self.points = pts
        self.tangents = _tangents(pts)
        self.m1 = _rmf(pts, self.tangents)
        self.m2 = np.stack([np.cross(t, m) for t, m in zip(self.tangents, self.m1)])
        seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        self.cum_s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum_s[-1])

    def project(self, p: np.ndarray) -> Projection:
        """Project a world point onto the tube-intrinsic frame.

        ponytail: linear scan over segments, O(N) per query. Fine for the
        centerlines we handle (hundreds of points); swap for a KD-tree /
        arc-length bucketing if narrowphase batching ever needs it.
        """
        p = np.asarray(p, dtype=float)
        a = self.points[:-1]
        b = self.points[1:]
        ab = b - a
        denom = np.einsum("ij,ij->i", ab, ab)
        denom[denom == 0] = 1.0
        u = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0.0, 1.0)
        foot = a + u[:, None] * ab
        d2 = np.einsum("ij,ij->i", p - foot, p - foot)
        j = int(np.argmin(d2))
        uj = float(u[j])

        s = float(self.cum_s[j] + uj * np.linalg.norm(ab[j]))
        t = _unit(self.tangents[j] + uj * (self.tangents[j + 1] - self.tangents[j]))
        radial = (p - foot[j]) - np.dot(p - foot[j], t) * t
        r = float(np.linalg.norm(radial))
        e_r = _unit(radial) if r > 0 else self.m1[j]
        # reference axes at the foot, re-orthogonalised against the local tangent
        m1 = _unit(self.m1[j] - np.dot(self.m1[j], t) * t)
        m2 = np.cross(t, m1)
        theta = float(np.arctan2(np.dot(radial, m2), np.dot(radial, m1)))
        return Projection(s=s, theta=theta, r=r, e_r=e_r, edge_param=uj)
