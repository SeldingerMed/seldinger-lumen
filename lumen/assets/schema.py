"""The asset schema -- the single integration seam (doc §5).

A *case* is a centerline graph plus a lumen field plus a device spawn, in an
explicitly-declared coordinate frame. Both ends emit this same object:

  * the open ``lumen.assets.procedural`` generator (provenance = "procedural")
  * a private patient pipeline in seldinger-ml (provenance = "patient(private)")

The explicit frame/spacing/origin is deliberate: it stops the open solver and a
patient pipeline from ever silently disagreeing on coordinates (the
voxel_scaled-vs-world-affine trap). Patient-derived assets MUST NOT live in this
open repository; the firewall check enforces provenance == "procedural" here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

SCHEMA_VERSION = "lumen-asset/0"


@dataclass
class Frame:
    name: str = "voxel_scaled"            # declared coordinate convention
    spacing_mm: tuple = (1.0, 1.0, 1.0)
    origin_mm: tuple = (0.0, 0.0, 0.0)


@dataclass
class Node:
    id: str
    position_mm: tuple


@dataclass
class Edge:
    id: str
    node_a: str
    node_b: str
    centerline_mm: list                    # list of [x, y, z]
    # lumen field for this edge, sampled on (s_grid x theta_grid):
    s_grid: list
    theta_grid: list
    R: list                                # shape (len(s_grid), len(theta_grid))
    fiber_dir: list = field(default_factory=list)   # optional HGO fiber dirs


@dataclass
class WallParams:
    # generic anisotropic-shell defaults; real calibration stays private
    C10: float = 1.0e4
    k1: float = 1.0e4
    k2: float = 1.0
    thickness_mm: float = 0.3


@dataclass
class DeviceSpawn:
    node_id: str
    insertion_mm: float = 0.0


@dataclass
class Asset:
    frame: Frame
    nodes: list                            # list[Node]
    edges: list                            # list[Edge]
    device_spawn: DeviceSpawn
    wall: WallParams = field(default_factory=WallParams)
    provenance: str = "procedural"         # "procedural" | "patient(private)"
    version: str = SCHEMA_VERSION

    # --- I/O -------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Asset":
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "Asset":
        return cls(
            frame=Frame(**d["frame"]),
            nodes=[Node(**n) for n in d["nodes"]],
            edges=[Edge(**e) for e in d["edges"]],
            device_spawn=DeviceSpawn(**d["device_spawn"]),
            wall=WallParams(**d.get("wall", {})),
            provenance=d.get("provenance", "procedural"),
            version=d.get("version", SCHEMA_VERSION),
        )

    # --- bridges to the core types --------------------------------------
    def edge_arrays(self, edge: Edge):
        """Return (centerline points, LumenField) for an edge."""
        from lumen.core.lumen_field import LumenField
        pts = np.asarray(edge.centerline_mm, dtype=float)
        lf = LumenField(np.asarray(edge.s_grid), np.asarray(edge.theta_grid),
                        np.asarray(edge.R))
        return pts, lf
