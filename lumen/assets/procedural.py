"""Synthetic anatomy generator (no patient data, ever).

Procedural tubes and bifurcations for examples, tests, and benchmarks. This is
the *only* sanctioned source of geometry in the open repo; every asset it emits
is tagged ``provenance="procedural"`` so the firewall check passes.

These are deliberately modality-neutral shapes -- a "tube" is a vessel, an
airway, or a bowel segment depending only on the radius and the profile you ask
for, not on anything in this module.
"""

from __future__ import annotations

import numpy as np

from lumen.assets.schema import Asset, DeviceSpawn, Edge, Frame, Node


def _edge_from_polyline(edge_id, a, b, pts, lf) -> Edge:
    return Edge(
        id=edge_id, node_a=a, node_b=b,
        centerline_mm=np.asarray(pts, dtype=float).tolist(),
        s_grid=lf.s.tolist(), theta_grid=lf.theta.tolist(), R=lf.R.tolist(),
    )


def straight_tube(length: float = 100.0, radius: float = 2.0, n: int = 64,
                  axis=(0.0, 0.0, 1.0)) -> Asset:
    """A single straight tube of constant radius."""
    from lumen.core.lumen_field import LumenField
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    t = np.linspace(0.0, length, n)
    pts = t[:, None] * axis[None, :]
    lf = LumenField.cylinder(length, radius, n=n)
    return Asset(
        frame=Frame(),
        nodes=[Node("n0", tuple(pts[0])), Node("n1", tuple(pts[-1]))],
        edges=[_edge_from_polyline("e0", "n0", "n1", pts, lf)],
        device_spawn=DeviceSpawn(node_id="n0"),
    )


def stenotic_tube(length: float = 100.0, radius: float = 2.0,
                  severity: float = 0.6, n: int = 96) -> Asset:
    """Straight tube with an axisymmetric narrowing at mid-length."""
    from lumen.core.lumen_field import LumenField
    t = np.linspace(0.0, length, n)
    pts = np.stack([np.zeros(n), np.zeros(n), t], axis=1)
    lf = LumenField.stenosis(length, radius, at=length / 2, severity=severity, n=n)
    a = straight_tube(length, radius, n)
    a.edges = [_edge_from_polyline("e0", "n0", "n1", pts, lf)]
    return a


def bifurcation(trunk: float = 50.0, branch: float = 50.0, radius: float = 2.0,
                angle_deg: float = 35.0, n: int = 48) -> Asset:
    """A Y: one trunk splitting into two branches.

    ponytail: branch-point blending of the lumen field is deferred (doc §3.5.2
    blends R near bifurcations). P0 stores the three edges meeting at a node;
    overlap blending lands when contact narrowphase needs it.
    """
    from lumen.core.lumen_field import LumenField
    ang = np.radians(angle_deg)
    zt = np.linspace(0.0, trunk, n)
    trunk_pts = np.stack([np.zeros(n), np.zeros(n), zt], axis=1)
    apex = trunk_pts[-1]
    sb = np.linspace(0.0, branch, n)
    left = apex + sb[:, None] * np.array([-np.sin(ang), 0.0, np.cos(ang)])
    right = apex + sb[:, None] * np.array([np.sin(ang), 0.0, np.cos(ang)])
    lf_t = LumenField.cylinder(trunk, radius, n=n)
    lf_b = LumenField.cylinder(branch, radius * 0.8, n=n)
    return Asset(
        frame=Frame(),
        nodes=[Node("trunk_in", tuple(trunk_pts[0])), Node("apex", tuple(apex)),
               Node("left_out", tuple(left[-1])), Node("right_out", tuple(right[-1]))],
        edges=[
            _edge_from_polyline("trunk", "trunk_in", "apex", trunk_pts, lf_t),
            _edge_from_polyline("left", "apex", "left_out", left, lf_b),
            _edge_from_polyline("right", "apex", "right_out", right, lf_b),
        ],
        device_spawn=DeviceSpawn(node_id="trunk_in"),
    )
