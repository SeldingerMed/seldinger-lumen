"""Anatomically proportioned aortic arch for the X-series clips (femoral approach).

The library's `procedural.aortic_arch_tree()` is a compact demo graph entered from an
'inlet' at the aortic root. For the hero and safety clips we want the clinical
configuration: the wire comes up the descending thoracic aorta, crosses the arch and
selects a supra-aortic branch. This builds that graph from the same procedural
primitives (Bezier centerlines + per-point lumen radii). Dimensions are typical adult
values (type I arch): ascending ~27 mm diameter, arch ~23 mm, descending ~20 mm,
brachiocephalic ~11 mm, left common carotid ~7 mm, left subclavian ~8.5 mm. They are
illustrative, not patient-derived.
"""

from __future__ import annotations

import numpy as np

from lumen.assets.procedural import _bezier, _edge_from_polyline, _lumen_from_radii
from lumen.assets.schema import Asset, DeviceSpawn, Frame, Node

# z = cranial, x = patient left, y = posterior (mm)
P = {
    "desc_in": (22.0, 30.0, -70.0),
    "lsa_j": (18.0, 20.0, 32.0),
    "lcca_j": (4.0, 8.0, 44.0),
    "bca_j": (-12.0, -4.0, 40.0),
    "root": (-12.0, -18.0, -30.0),
    "lsa_out": (45.0, 22.0, 85.0),
    "lcca_out": (8.0, 6.0, 120.0),
    "bca_out": (-28.0, -2.0, 95.0),
}

EDGES = [
    # id, a, b, ctrl1, ctrl2, r_start, r_end
    ("descending", "desc_in", "lsa_j", (22, 30, -30), (22, 28, 10), 10.0, 10.8),
    ("arch_distal", "lsa_j", "lcca_j", (14, 16, 40), (9, 12, 44), 11.0, 11.4),
    ("arch_proximal", "lcca_j", "bca_j", (-1, 4, 46), (-7, 0, 45), 11.5, 12.5),
    ("ascending", "bca_j", "root", (-20, -12, 30), (-16, -18, 0), 13.0, 13.8),
    ("left_subclavian", "lsa_j", "lsa_out", (22, 22, 52), (34, 22, 70), 4.3, 3.7),
    ("left_common_carotid", "lcca_j", "lcca_out", (4, 8, 64), (8, 7, 90), 3.6, 3.1),
    ("brachiocephalic", "bca_j", "bca_out", (-16, -4, 60), (-24, -3, 78), 5.6, 4.7),
]


def femoral_arch(scale: float = 1.0, n: int = 56) -> Asset:
    nodes = [Node(k, tuple(scale * np.asarray(v, float))) for k, v in P.items()]
    edges = []
    for eid, a, b, c1, c2, r0, r1 in EDGES:
        pts = _bezier(scale * np.asarray(P[a]), scale * np.asarray(c1, float),
                      scale * np.asarray(c2, float), scale * np.asarray(P[b]), n)
        radii = scale * np.linspace(r0, r1, n)
        edges.append(_edge_from_polyline(eid, a, b, pts, _lumen_from_radii(pts, radii)))
    return Asset(frame=Frame(), nodes=nodes, edges=edges,
                 device_spawn=DeviceSpawn(node_id="desc_in"))


TARGETS = {"lcca": "lcca_out", "bca": "bca_out", "lsa": "lsa_out"}
