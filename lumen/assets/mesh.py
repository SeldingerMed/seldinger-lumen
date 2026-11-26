"""Surface meshes: simplify, store compactly, read back.

Scan surfaces run to hundreds of thousands of triangles. For drawing and for shipping, a
vertex-clustering simplification on a fine grid keeps the look (sub-millimetre detail is far
below what any view shows) at a fraction of the size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

QUANTUM_MM = 0.01        # stored vertex precision


def weld(tris: np.ndarray, tol_mm: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Triangle soup ``(n, 3, 3)`` -> shared ``(vertices, faces)``."""
    v = np.asarray(tris, np.float64).reshape(-1, 3)
    key = np.round(v / tol_mm).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return v[first].astype(np.float32), inv.reshape(-1, 3).astype(np.uint32)


def simplify(verts: np.ndarray, faces: np.ndarray, cell_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: merge every vertex in a ``cell_mm`` cube to their mean, drop the
    triangles that collapse, and drop duplicates."""
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    if cell_mm <= 0 or not len(f):
        return v.astype(np.float32), f.astype(np.uint32)
    cell = np.floor((v - v.min(0)) / cell_mm).astype(np.int64)
    _, cid = np.unique(cell, axis=0, return_inverse=True)
    cid = cid.ravel()
    n = int(cid.max()) + 1
    sums = np.zeros((n, 3))
    np.add.at(sums, cid, v)
    counts = np.bincount(cid, minlength=n)[:, None]
    nv = sums / np.maximum(counts, 1)
    nf = cid[f]
    keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf = nf[keep]
    if len(nf):
        order = np.sort(nf, axis=1)
        _, first = np.unique(order, axis=0, return_index=True)
        nf = nf[np.sort(first)]
    used = np.unique(nf)
    remap = -np.ones(n, np.int64)
    remap[used] = np.arange(len(used))
    return nv[used].astype(np.float32), remap[nf].astype(np.uint32)


def simplify_to(verts, faces, max_triangles: int, start_mm: float = 0.2):
    """Simplify with a growing cell until at most ``max_triangles`` remain."""
    v, f, cell = np.asarray(verts), np.asarray(faces), start_mm
    while len(f) > max_triangles and cell < 10.0:
        v, f = simplify(verts, faces, cell)
        cell *= 1.3
    return v, f


def save_npz(path: str | Path, verts: np.ndarray, faces: np.ndarray) -> Path:
    """Compact storage: vertices as 0.01 mm integers, faces as small deltas."""
    v = np.asarray(verts, np.float64)
    origin = v.min(0) if len(v) else np.zeros(3)
    q = np.round((v - origin) / QUANTUM_MM)
    if q.size and q.max() > np.iinfo(np.uint32).max:
        raise ValueError("surface too large to store")
    f = np.asarray(faces, np.int64)
    p = Path(path)
    np.savez_compressed(p, origin=origin.astype(np.float64), quantum=np.float64(QUANTUM_MM),
                        vertices=q.astype(np.uint32),
                        faces=np.diff(f.ravel(), prepend=0).astype(np.int32))
    return p


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(path)) as z:
        verts = z["origin"] + z["vertices"].astype(np.float64) * float(z["quantum"])
        faces = np.cumsum(z["faces"].astype(np.int64)).reshape(-1, 3)
    if len(faces) and (faces.min() < 0 or faces.max() >= len(verts)):
        raise ValueError(f"{path}: surface faces point outside its vertices")
    return verts.astype(np.float32), faces.astype(np.uint32)
