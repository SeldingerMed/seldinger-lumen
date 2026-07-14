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


MAX_STENOSIS_SEVERITY_EXCLUSIVE = 0.9


def _arclength(pts: np.ndarray) -> np.ndarray:
    """Compute cumulative arclength along a polyline."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return np.array([])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else np.array([])
    return np.concatenate([[0.0], np.cumsum(seg)])


def _lumen_from_radii(pts: np.ndarray, radii: np.ndarray):
    from lumen.core.lumen_field import LumenField
    pts = np.asarray(pts, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(radii) != len(pts):
        raise ValueError("radii must have one value per centerline point")
    if np.any(radii <= 0.0):
        raise ValueError("radii must be positive")
    return LumenField(_arclength(pts), np.array([0.0]), radii[:, None])


def _bezier(p0, p1, p2, p3, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(n))[:, None]
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p0, p1, p2, p3))
    return ((1.0 - t) ** 3 * p0
            + 3.0 * (1.0 - t) ** 2 * t * p1
            + 3.0 * (1.0 - t) * t ** 2 * p2
            + t ** 3 * p3)


def _validate_demo_geometry(n: int, radius: float, *,
                            severity: float | None = None,
                            severity_name: str = "severity") -> int:
    n = int(n)
    if n < 8:
        raise ValueError("n must be >= 8")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if severity is not None and not (0.0 <= severity < MAX_STENOSIS_SEVERITY_EXCLUSIVE):
        raise ValueError(
            f"{severity_name} must be in [0, {MAX_STENOSIS_SEVERITY_EXCLUSIVE})"
        )
    return n


def _validate_fraction(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]")


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


def tortuous_tube(length: float = 100.0, radius: float = 2.4,
                  severity: float = 0.35, n: int = 96,
                  dilation: float = 0.16) -> Asset:
    """Curved, tapered single-vessel demo tube with focal dilation and narrowing."""
    n = _validate_demo_geometry(n, radius, severity=severity)
    if length <= 0.0:
        raise ValueError("length must be positive")
    _validate_fraction("dilation", dilation)
    pts = _bezier([0.0, 0.0, 0.0],
                  [-12.0, 2.0, 0.30 * length],
                  [16.0, -3.0, 0.68 * length],
                  [5.0, 1.0, length],
                  n)
    u = np.linspace(0.0, 1.0, n)
    taper = 1.0 - 0.16 * u
    bulge = 1.0 + float(dilation) * np.exp(-0.5 * ((u - 0.30) / 0.10) ** 2)
    lesion = severity * np.exp(-0.5 * ((u - 0.58) / 0.12) ** 2)
    ripple = 1.0 + 0.018 * np.sin(4.0 * np.pi * u)
    radii = radius * taper * bulge * ripple * (1.0 - lesion)
    lf = _lumen_from_radii(pts, radii)
    return Asset(
        frame=Frame(),
        nodes=[Node("n0", tuple(pts[0])), Node("n1", tuple(pts[-1]))],
        edges=[_edge_from_polyline("e0", "n0", "n1", pts, lf)],
        device_spawn=DeviceSpawn(node_id="n0"),
    )


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


def tortuous_tree(radius: float = 4.0, n: int = 44,
                  stenosis_severity: float = 0.30,
                  side_dilation: float = 0.28) -> Asset:
    """Asymmetric, curved multi-branch synthetic vessel for demos.

    This stays fully procedural, but it is intentionally less toy-like than the
    canonical Y: a curved trunk, a side branch, asymmetric daughter vessels,
    tapering radii, and a focal narrowing on one branch. It is meant for product
    demos and CV/rendering smoke tests; benchmark scenes remain the small canonical
    tube/stenosis/Y tasks.
    """
    n = _validate_demo_geometry(
        n, radius, severity=stenosis_severity, severity_name="stenosis_severity",
    )
    _validate_fraction("side_dilation", side_dilation)

    inlet = np.array([0.0, 0.0, 0.0])
    side = np.array([5.0, 2.0, 42.0])
    apex = np.array([13.0, -2.0, 78.0])
    side_out = np.array([-32.0, 4.0, 62.0])
    left_out = np.array([-26.0, -3.0, 112.0])
    right_out = np.array([45.0, 5.0, 109.0])

    trunk0 = _bezier(inlet, [-7.0, 1.0, 12.0], [10.0, 4.0, 27.0], side, n)
    trunk1 = _bezier(side, [0.0, -4.0, 54.0], [24.0, -2.0, 65.0], apex, n)
    side_branch = _bezier(side, [-8.0, 5.0, 45.0], [-24.0, 7.0, 52.0], side_out, n)
    left = _bezier(apex, [2.0, -7.0, 89.0], [-17.0, -5.0, 100.0], left_out, n)
    right = _bezier(apex, [24.0, 3.0, 88.0], [41.0, 8.0, 98.0], right_out, n)

    u = np.linspace(0.0, 1.0, n)
    trunk0_r = radius * (1.0 - 0.10 * u) * (1.0 + 0.025 * np.sin(2.0 * np.pi * u))
    trunk1_r = radius * (0.92 - 0.10 * u) * (1.0 + 0.020 * np.cos(2.0 * np.pi * u))
    side_bulge = 1.0 + float(side_dilation) * np.exp(-0.5 * ((u - 0.48) / 0.13) ** 2)
    side_r = radius * (0.62 - 0.10 * u) * side_bulge
    left_r = radius * (0.70 - 0.12 * u)
    right_base = radius * (0.76 - 0.10 * u)
    lesion = stenosis_severity * np.exp(-0.5 * ((u - 0.55) / 0.13) ** 2)
    right_r = right_base * (1.0 - lesion)

    nodes = [
        Node("inlet", tuple(inlet)),
        Node("side_junction", tuple(side)),
        Node("apex", tuple(apex)),
        Node("side_out", tuple(side_out)),
        Node("left_out", tuple(left_out)),
        Node("right_out", tuple(right_out)),
    ]
    edges = [
        _edge_from_polyline("trunk_in", "inlet", "side_junction",
                            trunk0, _lumen_from_radii(trunk0, trunk0_r)),
        _edge_from_polyline("trunk_mid", "side_junction", "apex",
                            trunk1, _lumen_from_radii(trunk1, trunk1_r)),
        _edge_from_polyline("side", "side_junction", "side_out",
                            side_branch, _lumen_from_radii(side_branch, side_r)),
        _edge_from_polyline("left", "apex", "left_out",
                            left, _lumen_from_radii(left, left_r)),
        _edge_from_polyline("right_stenotic", "apex", "right_out",
                            right, _lumen_from_radii(right, right_r)),
    ]
    return Asset(
        frame=Frame(),
        nodes=nodes,
        edges=edges,
        device_spawn=DeviceSpawn(node_id="inlet"),
    )


def aortic_arch_tree(radius: float = 5.0, n: int = 48) -> Asset:
    """Open procedural arch with supra-aortic branches."""
    n = _validate_demo_geometry(n, radius)

    inlet = np.array([0.0, 0.0, 0.0])
    arch_prox = np.array([20.0, -2.0, 34.0])
    arch_mid = np.array([14.0, 0.0, 72.0])
    arch_dist = np.array([-18.0, 2.0, 92.0])
    descending_out = np.array([-22.0, -2.0, 138.0])
    brachio_out = np.array([55.0, 10.0, 78.0])
    carotid_out = np.array([20.0, 12.0, 114.0])
    subclavian_out = np.array([-34.0, 8.0, 116.0])

    ascending = _bezier(inlet, [12.0, -8.0, 10.0], [28.0, -7.0, 22.0], arch_prox, n)
    arch_a = _bezier(arch_prox, [34.0, -4.0, 46.0], [31.0, 0.0, 64.0], arch_mid, n)
    arch_b = _bezier(arch_mid, [0.0, 3.0, 82.0], [-13.0, 4.0, 86.0], arch_dist, n)
    descending = _bezier(arch_dist, [-30.0, 2.0, 106.0], [-18.0, -3.0, 126.0],
                         descending_out, n)
    brachio = _bezier(arch_prox, [31.0, 8.0, 52.0], [47.0, 11.0, 62.0], brachio_out, n)
    carotid = _bezier(arch_mid, [19.0, 9.0, 86.0], [23.0, 12.0, 102.0], carotid_out, n)
    subclavian = _bezier(arch_dist, [-24.0, 9.0, 98.0], [-34.0, 9.0, 106.0],
                         subclavian_out, n)

    u = np.linspace(0.0, 1.0, n)
    main_r0 = radius * (1.04 - 0.06 * u)
    main_r1 = radius * (0.98 - 0.08 * u) * (1.0 + 0.035 * np.sin(np.pi * u))
    main_r2 = radius * (0.88 - 0.09 * u)
    main_r3 = radius * (0.78 - 0.14 * u)
    branch_r = radius * (0.52 - 0.16 * u)
    carotid_r = radius * (0.46 - 0.12 * u)
    subclavian_r = radius * (0.48 - 0.13 * u)

    nodes = [
        Node("inlet", tuple(inlet)),
        Node("arch_prox", tuple(arch_prox)),
        Node("arch_mid", tuple(arch_mid)),
        Node("arch_dist", tuple(arch_dist)),
        Node("descending_out", tuple(descending_out)),
        Node("brachio_out", tuple(brachio_out)),
        Node("carotid_out", tuple(carotid_out)),
        Node("subclavian_out", tuple(subclavian_out)),
    ]
    edges = [
        _edge_from_polyline("ascending", "inlet", "arch_prox",
                            ascending, _lumen_from_radii(ascending, main_r0)),
        _edge_from_polyline("arch_proximal", "arch_prox", "arch_mid",
                            arch_a, _lumen_from_radii(arch_a, main_r1)),
        _edge_from_polyline("arch_distal", "arch_mid", "arch_dist",
                            arch_b, _lumen_from_radii(arch_b, main_r2)),
        _edge_from_polyline("descending", "arch_dist", "descending_out",
                            descending, _lumen_from_radii(descending, main_r3)),
        _edge_from_polyline("brachiocephalic", "arch_prox", "brachio_out",
                            brachio, _lumen_from_radii(brachio, branch_r)),
        _edge_from_polyline("left_carotid", "arch_mid", "carotid_out",
                            carotid, _lumen_from_radii(carotid, carotid_r)),
        _edge_from_polyline("left_subclavian", "arch_dist", "subclavian_out",
                            subclavian, _lumen_from_radii(subclavian, subclavian_r)),
    ]
    return Asset(
        frame=Frame(),
        nodes=nodes,
        edges=edges,
        device_spawn=DeviceSpawn(node_id="inlet"),
    )
