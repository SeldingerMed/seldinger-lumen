"""Import segmented cross-sectional anatomy into Lumen assets.

This module is deliberately dependency-light. The stable seam is:

    volume -> binary mask -> centerline/radius graph -> Asset

Heavy DICOM readers and ML segmenters can sit in optional wrappers, but the core
conversion path remains NumPy-only so imported masks are easy to test and replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lumen.assets.schema import Asset, DeviceSpawn, Edge, Frame, Node


@dataclass
class Volume:
    data: np.ndarray
    spacing_mm: tuple = (1.0, 1.0, 1.0)
    origin_mm: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        self.data = np.asarray(self.data)
        if self.data.ndim != 3:
            raise ValueError(f"volume must be 3-D, got shape {self.data.shape}")
        self.spacing_mm = _triple(self.spacing_mm, "spacing_mm")
        self.origin_mm = _triple(self.origin_mm, "origin_mm")
        if any(s <= 0 for s in self.spacing_mm):
            raise ValueError(f"spacing_mm values must be positive, got {self.spacing_mm}")

    @property
    def mask(self):
        return np.asarray(self.data, dtype=bool)


def _triple(value, name: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if len(arr) != 3:
        raise ValueError(f"{name} must have three values")
    return tuple(float(x) for x in arr)


def load_npz_volume(path) -> Volume:
    """Load a 3-D ``.npz`` volume or mask.

    Expected arrays:
    - ``volume`` for raw intensity, or ``mask`` for an existing segmentation.
    - optional ``spacing_mm`` and ``origin_mm`` triples.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        if "volume" in data:
            arr = data["volume"]
        elif "mask" in data:
            arr = data["mask"].astype(bool)
        else:
            raise ValueError("npz must contain a 'volume' or 'mask' array")
        spacing = data["spacing_mm"] if "spacing_mm" in data else (1.0, 1.0, 1.0)
        origin = data["origin_mm"] if "origin_mm" in data else (0.0, 0.0, 0.0)
    return Volume(arr, spacing_mm=spacing, origin_mm=origin)


def load_dicom_series(path) -> Volume:
    """Load a DICOM series with SimpleITK when the optional imaging extra is present."""
    try:
        import SimpleITK as sitk
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("load_dicom_series requires the optional SimpleITK dependency") from e

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(path))
    if not series_ids:
        raise ValueError(f"no DICOM series found under {path!r}")
    files = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
    reader.SetFileNames(files)
    image = reader.Execute()
    direction = np.asarray(image.GetDirection(), float).reshape(3, 3)
    if not np.allclose(direction, np.eye(3), atol=1e-6):
        raise ValueError("DICOM direction cosines are not identity; resample to an "
                         "axis-aligned volume before importing into the current Asset schema")
    # SimpleITK arrays are z,y,x; Lumen volumes use x,y,z.
    data = np.asarray(sitk.GetArrayFromImage(image)).transpose(2, 1, 0)
    return Volume(data, spacing_mm=image.GetSpacing(), origin_mm=image.GetOrigin())


def segment_threshold(volume: Volume, threshold: float, foreground: str = "above") -> Volume:
    """Segment a volume by thresholding.

    This is a baseline backend, not a clinical segmenter. It is useful for synthetic
    CTA/MRA/airway fixtures and as the common output shape for model-backed segmenters.
    """
    if foreground == "above":
        mask = np.asarray(volume.data) >= float(threshold)
    elif foreground == "below":
        mask = np.asarray(volume.data) <= float(threshold)
    else:
        raise ValueError("foreground must be 'above' or 'below'")
    return Volume(mask, spacing_mm=volume.spacing_mm, origin_mm=volume.origin_mm)


@dataclass
class _SliceComponent:
    id: int
    z: int
    pixels: frozenset
    center_vox: np.ndarray
    center_mm: np.ndarray
    radius_mm: float


def asset_from_mask(mask, spacing_mm=(1.0, 1.0, 1.0), origin_mm=(0.0, 0.0, 0.0),
                    min_component_voxels: int = 4,
                    provenance: str = "segmented(imported)") -> Asset:
    """Convert an axial binary vessel/airway mask into a Lumen ``Asset``.

    The extractor is intentionally conservative: it follows connected components in
    adjacent axial slices, estimates each cross-section's equivalent circular radius,
    and compresses degree-2 runs into graph edges. It is a good import seam for
    already-segmented demo volumes; arbitrary tortuous clinical centerlines can later
    replace this extractor behind the same ``Asset`` boundary.
    """
    vol = Volume(np.asarray(mask, dtype=bool), spacing_mm=spacing_mm, origin_mm=origin_mm)
    if min_component_voxels <= 0:
        raise ValueError("min_component_voxels must be positive")
    comps = _slice_components(vol, min_component_voxels)
    if not comps:
        raise ValueError("mask contains no components large enough to import")
    links = _link_components(comps, vol.spacing_mm)
    return _components_to_asset(comps, links, vol.spacing_mm, vol.origin_mm, provenance)


def _slice_components(vol: Volume, min_voxels: int) -> list[_SliceComponent]:
    comps: list[_SliceComponent] = []
    spacing = np.asarray(vol.spacing_mm, float)
    origin = np.asarray(vol.origin_mm, float)
    next_id = 0
    for z in range(vol.data.shape[2]):
        labels = _connected_components_2d(vol.data[:, :, z])
        for pixels in labels:
            if len(pixels) < min_voxels:
                continue
            arr = np.asarray(list(pixels), dtype=float)
            xy = arr.mean(axis=0)
            center_vox = np.array([xy[0], xy[1], float(z)])
            center_mm = origin + center_vox * spacing
            area_mm2 = len(pixels) * spacing[0] * spacing[1]
            radius_mm = max(float(np.sqrt(area_mm2 / np.pi)), 0.5 * min(spacing[:2]))
            comps.append(_SliceComponent(next_id, z, pixels, center_vox, center_mm, radius_mm))
            next_id += 1
    return comps


def _connected_components_2d(mask2d) -> list[frozenset]:
    mask2d = np.asarray(mask2d, bool)
    seen = np.zeros(mask2d.shape, dtype=bool)
    out = []
    nx, ny = mask2d.shape
    for seed in np.argwhere(mask2d):
        sx, sy = int(seed[0]), int(seed[1])
        if seen[sx, sy]:
            continue
        stack = [(sx, sy)]
        seen[sx, sy] = True
        pixels = []
        while stack:
            x, y = stack.pop()
            pixels.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < nx and 0 <= yy < ny and mask2d[xx, yy] and not seen[xx, yy]:
                        seen[xx, yy] = True
                        stack.append((xx, yy))
        out.append(frozenset(pixels))
    return out


def _link_components(comps: list[_SliceComponent], spacing_mm) -> set[tuple[int, int]]:
    by_z: dict[int, list[_SliceComponent]] = {}
    for c in comps:
        by_z.setdefault(c.z, []).append(c)
    links = set()
    z_values = sorted(by_z)
    xy_spacing = np.asarray(spacing_mm[:2], float)
    for za, zb in zip(z_values[:-1], z_values[1:]):
        if zb != za + 1:
            continue
        for a in by_z[za]:
            candidates = []
            for b in by_z[zb]:
                overlap = bool(a.pixels & b.pixels)
                delta_xy = (a.center_vox[:2] - b.center_vox[:2]) * xy_spacing
                dist = float(np.linalg.norm(delta_xy))
                if overlap or dist <= a.radius_mm + b.radius_mm + max(xy_spacing):
                    candidates.append((dist, b.id))
            if candidates:
                for _, bid in sorted(candidates)[:2]:
                    links.add((a.id, bid))
    return links


def _components_to_asset(comps: list[_SliceComponent], links: set[tuple[int, int]],
                         spacing_mm, origin_mm, provenance: str) -> Asset:
    comp = {c.id: c for c in comps}
    adj: dict[int, set[int]] = {c.id: set() for c in comps}
    for a, b in links:
        adj[a].add(b)
        adj[b].add(a)
    endpoints = {c.id for c in comps if not adj[c.id]}
    min_z = min(c.z for c in comps)
    max_z = max(c.z for c in comps)
    endpoints |= {c.id for c in comps if c.z in (min_z, max_z)}
    specials = {cid for cid, ns in adj.items() if len(ns) != 2} | endpoints
    node_ids = {cid: f"n{cid}" for cid in sorted(specials)}
    nodes = [Node(node_ids[cid], tuple(comp[cid].center_mm)) for cid in sorted(specials)]

    edges: list[Edge] = []
    visited: set[frozenset] = set()
    for start in sorted(specials):
        for nxt in sorted(adj[start]):
            key = frozenset((start, nxt))
            if key in visited:
                continue
            path = [start]
            prev, cur = start, nxt
            visited.add(key)
            while cur not in specials:
                path.append(cur)
                choices = [n for n in sorted(adj[cur]) if n != prev]
                if not choices:
                    break
                prev, cur = cur, choices[0]
                visited.add(frozenset((prev, cur)))
            path.append(cur)
            if path[0] == path[-1] or path[-1] not in specials:
                continue
            edges.append(_edge_for_path(f"e{len(edges)}", node_ids[path[0]],
                                        node_ids[path[-1]], [comp[i] for i in path]))

    if not edges and len(comps) == 1:
        c = comps[0]
        p0 = c.center_mm - np.array([0.0, 0.0, 0.5 * spacing_mm[2]])
        p1 = c.center_mm + np.array([0.0, 0.0, 0.5 * spacing_mm[2]])
        nodes = [Node("n0", tuple(p0)), Node("n1", tuple(p1))]
        edges = [_edge_for_points("e0", "n0", "n1", [p0, p1], [c.radius_mm, c.radius_mm])]
        spawn = "n0"
    else:
        spawn_comp = min(specials, key=lambda cid: (comp[cid].z, comp[cid].center_mm[2]))
        spawn = node_ids[spawn_comp]

    return Asset(
        frame=Frame(name="world_mm", spacing_mm=spacing_mm, origin_mm=origin_mm),
        nodes=nodes,
        edges=edges,
        device_spawn=DeviceSpawn(node_id=spawn),
        provenance=provenance,
    )


def _edge_for_path(edge_id: str, node_a: str, node_b: str,
                   comps: list[_SliceComponent]) -> Edge:
    pts = [c.center_mm for c in comps]
    radii = [c.radius_mm for c in comps]
    return _edge_for_points(edge_id, node_a, node_b, pts, radii)


def _edge_for_points(edge_id: str, node_a: str, node_b: str, pts, radii) -> Edge:
    pts = np.asarray(pts, float)
    radii = np.asarray(radii, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else np.array([])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    theta = np.array([0.0])
    return Edge(
        id=edge_id,
        node_a=node_a,
        node_b=node_b,
        centerline_mm=pts.tolist(),
        s_grid=s.tolist(),
        theta_grid=theta.tolist(),
        R=radii[:, None].tolist(),
    )
