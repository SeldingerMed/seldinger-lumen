"""Turn a vessel surface (triangles) into a Lumen ``Asset`` the solver can navigate.

This is the geometry core behind ``lumen import``: a real scan is segmented, the vessel mask
becomes a closed surface (``lumen.assets.scan_import``), and this module converts that surface
into a centerline tree with a fitted wall radius R(s, theta) on every edge. It also accepts a
surface mesh directly (STL).

Pipeline
--------
1. Read the STL and scale it to millimetres.
2. Voxelize the lumen with exact z-scanline crossings. Two surface conventions:
   ``"shell"`` (a thin-walled tube: outer and inner surfaces with normals pointing out of the
   wall material; the lumen is bounded by the inner surface) and ``"solid"`` (a closed lumen
   surface with outward normals).
3. Extract a centerline tree from the voxelized lumen (TEASAR-style): a cost-weighted Dijkstra
   from the inlet with step cost ~ 1/d^2 (d = distance to the wall) keeps paths centred; branch
   endpoints are found as the geodesically farthest uncovered voxels.
4. Smooth each edge and re-centre it with exact rays cast against the mesh in the normal plane.
5. Fit the lumen field R(s, theta) on each edge by exact ray casting in Lumen's own
   rotation-minimizing frame (``CenterlineFrame``), so theta matches the solver's convention.
   Rays that escape through a branch ostium are clipped to the local cross-section.
6. Split edges at named target sites so the environment can address them as nodes.
7. Report the conversion error in both directions (source wall -> Lumen wall and Lumen wall ->
   source wall) in millimetres.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lumen.assets.schema import Asset, DeviceSpawn, Edge, Frame, Node

PROVENANCE = "imported(surface)"
N_THETA = 32


# ---------------------------------------------------------------------------------------------
# mesh I/O
# ---------------------------------------------------------------------------------------------
def read_stl(path: str | Path) -> np.ndarray:
    """Return triangles as an (F, 3, 3) float64 array (binary or ASCII STL)."""
    raw = Path(path).read_bytes()
    if len(raw) >= 84:
        n = struct.unpack("<I", raw[80:84])[0]
        if 84 + 50 * n == len(raw):
            rec = np.frombuffer(raw[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
            return rec["v"].astype(np.float64)
    verts = [list(map(float, ln.split()[1:4])) for ln in raw.decode("ascii", "ignore").splitlines()
             if ln.strip().startswith("vertex")]
    if not verts or len(verts) % 3:
        raise ValueError(f"not a readable STL: {path}")
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


def write_stl(path: str | Path, tris: np.ndarray) -> None:
    tris = np.asarray(tris, dtype=np.float32)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = nrm, tris
    Path(path).write_bytes(b"\0" * 80 + struct.pack("<I", len(tris)) + rec.tobytes())


def _weld(tris: np.ndarray, tol: float = 1e-4):
    """Face vertex indices with vertices merged on a ``tol`` grid (one packed int64 key each)."""
    q = np.round((tris.reshape(-1, 3) - tris.reshape(-1, 3).min(0)) / tol).astype(np.int64)
    if q.max() >= 2**21:
        raise ValueError("mesh too large for the packed weld key; raise tol")
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    return None, inv.reshape(-1, 3)


def lumen_side_cached(tris: np.ndarray, mode: str, cache: str | Path | None) -> np.ndarray:
    """``lumen_side`` with an optional on-disk cache keyed by the triangle bytes."""
    if cache is None:
        return lumen_side(tris, mode)
    import hashlib
    key = hashlib.sha256(np.ascontiguousarray(tris, dtype=np.float64).tobytes() + mode.encode()).hexdigest()[:16]
    f = Path(cache) / f"side-{key}.npy"
    if f.exists():
        return np.load(f)
    side = lumen_side(tris, mode)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, side)
    return side


def lumen_side(tris: np.ndarray, mode: str, crease_deg: float = 55.0) -> np.ndarray:
    """Per-face sign s: +1 if the normal points into the lumen, -1 if out of it, 0 if the face
    does not bound the lumen.

    ``solid``: every face bounds the lumen with an outward normal (-1).
    ``shell``: split the wall at creases; the two largest smooth patches are the outer and inner
    surfaces (inner = smaller area, normals into the lumen); the rest (end rings) bound no lumen.
    Small crease fragments inherit the label of the nearest large-patch face.
    """
    n = len(tris)
    if mode == "solid":
        return -np.ones(n)
    if mode != "shell":
        raise ValueError(f"mode must be 'shell' or 'solid', got {mode!r}")
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    _, F = _weld(tris)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm /= np.maximum(2 * area, 1e-18)[:, None]
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(n), 3)
    e.sort(axis=1)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, fid = e[order], fid[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    a, b = fid[:-1][same], fid[1:][same]
    smooth = np.einsum("ij,ij->i", nrm[a], nrm[b]) > np.cos(np.radians(crease_deg))
    G = sp.coo_matrix((np.ones(smooth.sum()), (a[smooth], b[smooth])), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    patch_area = np.bincount(lab, weights=area)
    big = np.argsort(-patch_area)[:2]
    # an open thin-walled shell has two large walls of similar area. One smooth patch, or a
    # second patch far smaller than the first (an end cap), is a closed surface of the lumen
    if len(big) < 2 or patch_area[big[1]] < 0.2 * patch_area[big[0]]:
        raise ValueError("this mesh does not look like an open thin-walled shell (no inner and outer "
                         "wall of similar size); for a closed surface of the lumen use mode='solid'")
    inner = big[1] if patch_area[big[0]] >= patch_area[big[1]] else big[0]   # the smaller wall
    side = np.zeros(n)
    side[lab == inner] = 1.0
    cen = tris.mean(axis=1)
    small = ~np.isin(lab, big)
    small_patches = np.unique(lab[small])
    keep_big = np.isin(lab, big)
    tree = cKDTree(cen[keep_big])
    big_side = side[keep_big]
    big_lab = lab[keep_big]
    for p in small_patches:
        f = np.where(lab == p)[0]
        if patch_area[p] > 0.002 * patch_area[big].sum():     # a real end ring: bounds no lumen
            continue
        _, j = tree.query(cen[f])
        side[f] = np.where(big_lab[j] == inner, big_side[j], 0.0)
    return side


# ---------------------------------------------------------------------------------------------
# voxelization
# ---------------------------------------------------------------------------------------------
@dataclass
class Grid:
    origin: np.ndarray      # centre of voxel (0, 0, 0), mm
    h: float
    occ: np.ndarray         # bool lumen occupancy (nx, ny, nz)

    def centers(self, idx: np.ndarray) -> np.ndarray:
        return self.origin + idx * self.h


def voxelize(tris: np.ndarray, side: np.ndarray, h: float, pad: int = 2) -> tuple[Grid, dict]:
    """Lumen occupancy from exact crossings of +z rays through voxel-column centres."""
    lo = tris.reshape(-1, 3).min(0) - pad * h
    hi = tris.reshape(-1, 3).max(0) + pad * h
    jitter = np.array([1.37e-4, 2.91e-4, 0.0]) * h       # keep column centres off mesh edges
    origin = lo + 0.5 * h + jitter
    shape = np.ceil((hi - lo) / h).astype(int)
    nx, ny, nz = shape
    use = np.flatnonzero(side != 0)
    cols, zs, ents = [], [], []
    for c0 in range(0, len(use), 20_000):          # bounded memory: triangles in chunks
        f = use[c0:c0 + 20_000]
        T = tris[f]
        sgn = side[f]
        nz_face = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])[:, 2]
        keep = np.abs(nz_face) > 1e-14
        T, sgn, nz_face = T[keep], sgn[keep], nz_face[keep]
        xy = T[:, :, :2]
        i0 = np.ceil((xy[:, :, 0].min(1) - origin[0]) / h).astype(np.int64)
        i1 = np.floor((xy[:, :, 0].max(1) - origin[0]) / h).astype(np.int64)
        j0 = np.ceil((xy[:, :, 1].min(1) - origin[1]) / h).astype(np.int64)
        j1 = np.floor((xy[:, :, 1].max(1) - origin[1]) / h).astype(np.int64)
        ni, nj = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
        cnt = ni * nj
        t = np.repeat(np.arange(len(T)), cnt)
        k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ci = i0[t] + k // nj[t]
        cj = j0[t] + k % nj[t]
        px = origin[0] + ci * h
        py = origin[1] + cj * h
        A, B, C = T[t, 0], T[t, 1], T[t, 2]
        d = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
        l1 = ((B[:, 1] - C[:, 1]) * (px - C[:, 0]) + (C[:, 0] - B[:, 0]) * (py - C[:, 1])) / d
        l2 = ((C[:, 1] - A[:, 1]) * (px - C[:, 0]) + (A[:, 0] - C[:, 0]) * (py - C[:, 1])) / d
        l3 = 1.0 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        cols.append(ci[inside] * ny + cj[inside])
        zs.append((l1 * A[:, 2] + l2 * B[:, 2] + l3 * C[:, 2])[inside])
        ents.append(((sgn * nz_face)[t] > 0)[inside])      # +z along a lumen-facing normal: entering
    col, z, enter = np.concatenate(cols), np.concatenate(zs), np.concatenate(ents)
    del cols, zs, ents
    order = np.lexsort((z, col))
    col, z, enter = col[order], z[order], enter[order]
    occ = np.zeros((nx, ny, nz), dtype=bool)
    bad = 0
    starts = np.flatnonzero(np.r_[True, col[1:] != col[:-1]])
    ends = np.r_[starts[1:], len(col)]
    for s0, s1 in zip(starts, ends):
        zz, ee = z[s0:s1], enter[s0:s1]
        c = col[s0]
        ii, jj = divmod(int(c), ny)
        open_at = None
        for zv, ev in zip(zz, ee):
            if ev:
                if open_at is not None:
                    bad += 1
                open_at = zv
            elif open_at is not None:
                k0 = int(np.ceil((open_at - origin[2]) / h))
                k1 = int(np.floor((zv - origin[2]) / h))
                if k1 >= k0:
                    occ[ii, jj, max(k0, 0):min(k1, nz - 1) + 1] = True
                open_at = None
            else:
                bad += 1
        if open_at is not None:
            bad += 1
    return Grid(origin, h, occ), {"columns_with_hits": int(len(starts)), "inconsistent_crossings": int(bad),
                                  "voxels": int(occ.sum()), "volume_mm3": float(occ.sum() * h**3)}


# ---------------------------------------------------------------------------------------------
# rays
# ---------------------------------------------------------------------------------------------
class Rays:
    """Exact first-hit ray casting against the lumen-bounding faces, bounded to ``far`` mm.

    Triangles are hashed into a uniform grid (each registered in every cell its bounding box,
    dilated by half a march step, overlaps). Rays march in ``step`` segments; a segment only tests
    the triangles of the cell holding its midpoint, which by the dilation contains every triangle
    the segment can cross. Intersections are exact (Moller-Trumbore), so the result does not
    depend on the cell size; memory stays proportional to the surface, not to ray length.
    """

    def __init__(self, tris: np.ndarray, side: np.ndarray, cell: float = 1.5, step: float = 1.0):
        T = np.asarray(tris[side != 0], float)
        self.v0, self.e1, self.e2 = T[:, 0], T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
        self.cell, self.step = float(cell), float(step)
        dil = 0.5 * step + 1e-6
        lo_t, hi_t = T.min(1) - dil, T.max(1) + dil
        self.lo = lo_t.min(0) - cell
        self.dims = np.ceil((hi_t.max(0) + cell - self.lo) / cell).astype(np.int64) + 1
        cids, tris_ = [], []
        for f0 in range(0, len(T), 20_000):
            sl = slice(f0, f0 + 20_000)
            c0 = np.floor((lo_t[sl] - self.lo) / cell).astype(np.int64)
            c1 = np.floor((hi_t[sl] - self.lo) / cell).astype(np.int64)
            n = c1 - c0 + 1
            cnt = n.prod(1)
            t = np.repeat(np.arange(len(c0)), cnt)
            k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            ny, nz = n[t, 1], n[t, 2]
            ix = c0[t, 0] + k // (ny * nz)
            iy = c0[t, 1] + (k // nz) % ny
            iz = c0[t, 2] + k % nz
            cids.append((ix * self.dims[1] + iy) * self.dims[2] + iz)
            tris_.append((t + f0).astype(np.int32))
        cid, t = np.concatenate(cids), np.concatenate(tris_)
        del cids, tris_
        order = np.argsort(cid, kind="stable")
        cid, self.tri = cid[order], t[order].astype(np.int32)
        self.cells, self.start = np.unique(cid, return_index=True)
        self.count = np.diff(np.r_[self.start, len(cid)])

    def first_hit(self, origins: np.ndarray, dirs: np.ndarray, far: float = 40.0, chunk: int = 256) -> np.ndarray:
        """Distance to the first surface hit along each ray (``far`` if none within ``far``)."""
        if len(origins) > chunk:
            return np.concatenate([self.first_hit(origins[i:i + chunk], dirs[i:i + chunk], far, chunk)
                                   for i in range(0, len(origins), chunk)])
        O = np.asarray(origins, float)
        D = np.asarray(dirs, float)
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        out = np.full(len(O), np.inf)
        active = np.arange(len(O))
        for k in range(int(np.ceil(far / self.step))):
            if len(active) == 0:
                break
            mid = O[active] + D[active] * ((k + 0.5) * self.step)
            c = np.floor((mid - self.lo) / self.cell).astype(np.int64)
            inb = np.all((c >= 0) & (c < self.dims), axis=1)
            cid = (c[:, 0] * self.dims[1] + c[:, 1]) * self.dims[2] + c[:, 2]
            pos = np.clip(np.searchsorted(self.cells, cid), 0, len(self.cells) - 1)
            ok = inb & (self.cells[pos] == cid)
            if not ok.any():
                continue
            rr = np.flatnonzero(ok)
            cnt = self.count[pos[rr]]
            ray = np.repeat(rr, cnt)
            j = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            tri = self.tri[np.repeat(self.start[pos[rr]], cnt) + j]
            a = active[ray]
            o, d = O[a], D[a]
            e1, e2 = self.e1[tri], self.e2[tri]
            pv = np.cross(d, e2)
            det = np.einsum("ij,ij->i", e1, pv)
            good = np.abs(det) > 1e-14
            inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
            tv = o - self.v0[tri]
            u = np.einsum("ij,ij->i", tv, pv) * inv
            qv = np.cross(tv, e1)
            v = np.einsum("ij,ij->i", d, qv) * inv
            tt = np.einsum("ij,ij->i", e2, qv) * inv
            hit = good & (u >= 0) & (v >= 0) & (u + v <= 1) & (tt > 1e-9) & (tt <= (k + 1) * self.step + 1e-9)
            if hit.any():
                np.minimum.at(out, a[hit], tt[hit])
            done = np.isfinite(out[active])
            active = active[~done]
        out[~np.isfinite(out)] = far
        return out


# ---------------------------------------------------------------------------------------------
# centerline tree
# ---------------------------------------------------------------------------------------------
@dataclass
class Skeleton:
    nodes: dict = field(default_factory=dict)     # name -> xyz (mm)
    edges: list = field(default_factory=list)     # (node_a, node_b, (n,3) polyline)
    info: dict = field(default_factory=dict)


def centerline_tree(grid: Grid, root_mm: np.ndarray, min_branch_mm: float = 4.0,
                    cover_scale: float = 1.6, collapse_mm: float = 2.0, max_iter: int = 2000) -> Skeleton:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    occ, h = grid.occ, grid.h
    idx = np.argwhere(occ)
    N = len(idx)
    vid = -np.ones(occ.shape, dtype=np.int32)
    vid[tuple(idx.T)] = np.arange(N, dtype=np.int32)
    # distance to wall: nearest non-lumen 6-neighbour among boundary voxels
    pad = np.pad(occ, 1)
    interior = pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] & pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1] \
        & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
    bnd = np.argwhere(occ & ~interior)
    dt = cKDTree(bnd * h).query(idx * h)[0] + 0.5 * h
    rows, cols = [], []
    for ax in range(3):
        sh = [0, 0, 0]
        sh[ax] = 1
        a = vid[: occ.shape[0] - sh[0], : occ.shape[1] - sh[1], : occ.shape[2] - sh[2]]
        b = vid[sh[0]:, sh[1]:, sh[2]:]
        m = (a >= 0) & (b >= 0)
        rows.append(a[m])
        cols.append(b[m])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    w = h * 0.5 * (1.0 / dt[rows] ** 2 + 1.0 / dt[cols] ** 2)
    W = sp.csr_matrix((w, (rows, cols)), shape=(N, N))
    del rows, cols, w
    pts = idx * h + grid.origin
    root = int(np.argmin(np.linalg.norm(pts - np.asarray(root_mm, float), axis=1)))
    _, pred = dijkstra(W, directed=False, indices=root, return_predecessors=True)
    geo = dijkstra(W, directed=False, indices=root, unweighted=True) * h
    del W
    reach = np.isfinite(geo)
    covered = ~reach                  # tree regions and rejected spurs: never a new endpoint
    owner = -np.ones(N, dtype=np.int64)   # tree voxel whose cover sphere holds this voxel
    par = -np.ones(N, dtype=np.int64)     # parent pointer of tree voxels (toward the inlet)
    on_tree = np.zeros(N, dtype=bool)
    on_tree[root] = True
    owner[root] = root
    shape = np.asarray(occ.shape)

    def cover(path, radius, own):
        for v, r in zip(path, radius):
            rv = int(np.ceil(r / h))
            c = idx[v]
            lo, hi = np.maximum(c - rv, 0), np.minimum(c + rv + 1, shape)
            sub = vid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            g = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            d2 = (g[0] - c[0]) ** 2 + (g[1] - c[1]) ** 2 + (g[2] - c[2]) ** 2
            ids = sub[(d2 <= (r / h) ** 2) & (sub >= 0)]
            covered[ids] = True
            if own:
                free = ids[owner[ids] < 0]
                owner[free] = v

    leaves, rejected, it = [], 0, 0
    while it < max_iter:
        it += 1
        cand = np.where(covered, -1.0, geo)
        c = int(np.argmax(cand))
        if cand[c] < 0:
            break
        # walk the cost-weighted predecessor chain until it enters an existing vessel's cover
        # region, then join the tree at the centerline voxel that owns that region (so sibling
        # branches never run parallel centerlines through their shared parent vessel)
        path = [c]
        while owner[path[-1]] < 0:
            p = pred[path[-1]]
            if p < 0:
                break
            path.append(int(p))
        join = int(owner[path[-1]]) if owner[path[-1]] >= 0 else root
        path = np.asarray(path[:-1] if len(path) > 1 else path)
        # a real branch is long compared with its own radius; the uncovered pocket of a wall bulge
        # or of a flat end cap's corner is about as long as it is deep
        L = float(np.linalg.norm(np.diff(pts[path], axis=0), axis=1).sum()) + h
        accept = (L >= max(min_branch_mm, 1.5 * float(dt[path].max()))) or not leaves
        if accept:
            if not leaves:                      # first path: the full chain to the inlet
                chain = [c]
                while chain[-1] != root:
                    chain.append(int(pred[chain[-1]]))
                path, join = np.asarray(chain[:-1]), root
            par[path[:-1]] = path[1:]
            par[path[-1]] = join
            on_tree[path] = True
            leaves.append(c)
            cover(path[::-1], cover_scale * dt[path[::-1]] + h, own=True)
        else:
            rejected += 1
            cover(path, cover_scale * dt[path] + h, own=False)
        covered[path] = True
    # tree topology from the parent pointers
    children: dict[int, list[int]] = {}
    for v in np.flatnonzero(on_tree):
        if v != root:
            children.setdefault(int(par[v]), []).append(int(v))
    tree_len = {root: 0.0}
    order = [root]
    for v in order:
        for ch in children.get(v, []):
            tree_len[ch] = tree_len[v] + float(np.linalg.norm(pts[ch] - pts[v]))
            order.append(ch)
    # junctions: branching voxels; one closer than ~a vessel radius (along the tree) to the
    # previous kept junction is merged into it (its child edges then start at that junction)
    name = {root: "inlet"}
    for k, leaf in enumerate(leaves):
        name[leaf] = f"end{k}"
    kept = {root: root}
    junction = []
    for v in order:
        if v == root:
            continue
        anc = kept[int(par[v])]
        if len(children.get(v, [])) >= 2 and anc != root and \
                tree_len[v] - tree_len[anc] < max(collapse_mm, 0.75 * dt[anc]):
            kept[v] = anc
        elif len(children.get(v, [])) >= 2:
            junction.append(v)
            kept[v] = v
        else:
            kept[v] = anc
    for k, v in enumerate(junction):
        name[v] = f"j{k}"
    sk = Skeleton()
    for v, nm in name.items():
        sk.nodes[nm] = pts[v].copy()
    for v in name:                     # walk up from every named node to the previous kept junction
        if v == root:
            continue
        chain = [v]
        u = v
        target = kept[int(par[v])]
        while u != target:
            u = int(par[u])
            chain.append(u)
        sk.edges.append((name[target], name[v], pts[np.asarray(chain[::-1])].copy()))
    sk.info = {"lumen_voxels": int(N), "reachable_voxels": int(reach.sum()), "leaves": len(leaves),
               "junctions": len(junction), "rejected_spurs": int(rejected), "iterations": it,
               "max_wall_distance_mm": float(dt.max())}
    return sk


# ---------------------------------------------------------------------------------------------
# smoothing, re-centring and the lumen field
# ---------------------------------------------------------------------------------------------
def resample(poly: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return poly[[0, -1]]
    n = max(int(np.ceil(s[-1] / step)) + 1, 2)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, poly[:, k]) for k in range(3)], axis=1)


def smooth(poly: np.ndarray, sigma_pts: float) -> np.ndarray:
    if len(poly) < 3 or sigma_pts <= 0:
        return poly
    r = min(int(np.ceil(3 * sigma_pts)), len(poly) - 2)   # the mirror padding needs r + 1 interior points
    if r < 1:
        return poly
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_pts) ** 2)
    k /= k.sum()
    padded = np.concatenate([2 * poly[:1] - poly[r:0:-1], poly, 2 * poly[-1:] - poly[-2:-r - 2:-1]])
    out = np.stack([np.convolve(padded[:, j], k, mode="valid") for j in range(3)], axis=1)
    out[0], out[-1] = poly[0], poly[-1]
    return out


def _normal_frame(poly: np.ndarray):
    from lumen.core.frame import CenterlineFrame
    fr = CenterlineFrame(poly)
    t = fr.tangents
    m1 = fr.m1 - np.einsum("ij,ij->i", fr.m1, t)[:, None] * t
    m1 /= np.linalg.norm(m1, axis=1, keepdims=True)
    m2 = np.cross(t, m1)
    return fr, t, m1, m2


def cross_section(rays: Rays, poly: np.ndarray, n_theta: int = N_THETA):
    _, t, m1, m2 = _normal_frame(poly)
    th = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    D = np.cos(th)[None, :, None] * m1[:, None, :] + np.sin(th)[None, :, None] * m2[:, None, :]
    O = np.repeat(poly[:, None, :], n_theta, axis=1)
    R = rays.first_hit(O.reshape(-1, 3), D.reshape(-1, 3)).reshape(len(poly), n_theta)
    return th, R, D


def _local_size(m: np.ndarray, half: int = 6) -> np.ndarray:
    """For each section, the median section size of the vessel either side of it."""
    n = len(m)
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        around = np.r_[m[lo:i], m[i + 1:hi]]
        out[i] = float(np.median(around)) if len(around) else float(m[i])
    return out


def along_vessel(R: np.ndarray, wide: float = 3.0, slack_mm: float = 1.0,
                 narrow: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Check each cross-section against the vessel either side of it. Returns (R, odd rows).

    A section cut along a neighbouring vessel, or through a gap in the segmentation, has every
    ray long, so its own median is long too and a per-section test passes it; a centre line
    running along the wall of a thin artery leaves rays near zero on one side and far too long
    on the other. Against the median of the sections around it (``ref``), a section whose
    median is over ``wide`` x ref + ``slack_mm`` or under ``narrow`` x ref (collapsed to almost
    nothing) is rebuilt from its neighbours, and no single ray may exceed wide x ref + slack.
    A real aneurysm or other bulge, two or three times the parent vessel, stays under that, and
    a genuinely wide stretch (an artery's origin from the aorta) raises its own ``ref``; a
    stenosis of up to about 85% in diameter stays above the collapse test."""
    R = np.asarray(R, float).copy()
    n = len(R)
    odd = np.zeros(n, bool)
    if n < 3:
        return R, odd
    m = np.median(R, axis=1)
    ref = _local_size(m)
    odd = (m > wide * ref + slack_mm) | (m < narrow * ref)
    if odd.all():                                    # nothing to judge by
        return R, np.zeros(n, bool)
    if odd.any():
        idx = np.arange(n)
        for j in range(R.shape[1]):
            R[odd, j] = np.interp(idx[odd], idx[~odd], R[~odd, j])
        ref = _local_size(np.median(R, axis=1))
    np.minimum(R, (wide * ref + slack_mm)[:, None], out=R)
    return R, odd


def fill_ostia(R: np.ndarray, factor: float = 1.25, far: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """Close a vessel's wall across side openings.

    A ray that leaves through a branch ostium (or an open end) measures the other vessel, not
    this one. Such rays (longer than ``factor`` x the section's median, or no hit within ``far``)
    are replaced by periodic interpolation in theta from the rays that hit this vessel's wall, so
    each edge's R(s, theta) is its own closed tube; the branch's own edge supplies the opening.
    Rows where every ray escaped are interpolated along s from valid rows, then each section is
    checked against the vessel either side of it (``along_vessel``). Returns (R, escaped mask).
    """
    R = np.asarray(R, float).copy()
    n_s, n_t = R.shape
    th = np.arange(n_t)
    esc = (R >= far - 1e-9) | (R > factor * np.median(R, axis=1, keepdims=True))
    row_ok = np.zeros(n_s, bool)
    for i in range(n_s):
        good = ~esc[i]
        if good.sum() >= max(3, n_t // 4):
            row_ok[i] = True
            if not good.all():
                R[i, ~good] = np.interp(th[~good], th[good], R[i, good], period=n_t)
    if not row_ok.any():
        raise ValueError("no cross-section of this edge hits its own wall")
    s_idx = np.arange(n_s)
    for j in range(n_t):
        R[~row_ok, j] = np.interp(s_idx[~row_ok], s_idx[row_ok], R[row_ok, j])
    esc[~row_ok] = True
    R, odd = along_vessel(R)
    esc[odd] = True
    return R, esc


def recentre(rays: Rays, poly: np.ndarray, iters: int = 3, keep_ends_mm: float = 0.0) -> np.ndarray:
    p = poly.copy()
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    for _ in range(iters):
        th, R, D = cross_section(rays, p)
        Rc, esc = fill_ostia(R)
        escaped = esc.any(axis=1)
        shift = np.mean(Rc[:, :, None] * D, axis=1)
        mov = ~escaped & (s > keep_ends_mm) & (s < s[-1] - keep_ends_mm)
        mov[0] = mov[-1] = False
        p[mov] += 0.7 * shift[mov]
    return p


# ---------------------------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------------------------
def _split_at(asset_edges: list, nodes: dict, site_name: str, site: np.ndarray):
    """Insert node ``site_name`` at the centerline vertex nearest ``site``."""
    best = None
    for ei, (a, b, poly) in enumerate(asset_edges):
        d = np.linalg.norm(poly - site, axis=1)
        k = int(np.argmin(d))
        if best is None or d[k] < best[0]:
            best = (d[k], ei, k)
    assert best is not None, "no edges to split"
    dist, ei, k = best
    a, b, poly = asset_edges[ei]
    if len(poly) == 2:                           # too short to split at a vertex: add its midpoint
        poly = np.vstack([poly[0], 0.5 * (poly[0] + poly[1]), poly[1]])
    k = int(np.clip(k, 1, len(poly) - 2))
    nodes[site_name] = poly[k].copy()
    asset_edges[ei] = (a, site_name, poly[: k + 1])
    asset_edges.insert(ei + 1, (site_name, b, poly[k:]))
    return float(dist)


def import_mesh(mesh_path: str | Path, *, scale_to_mm: float = 1000.0, mode: str = "shell",
                voxel_mm: float = 0.5, root_mm=None, sites: dict | None = None,
                min_branch_mm: float = 4.0, step_mm: float = 0.5, cache: str | Path | None = None
                ) -> tuple[Asset, dict]:
    """Return ``(asset, report)``. ``root_mm`` is the device inlet point; ``sites`` maps node
    names to points at which edges are split (e.g. navigation targets)."""
    tris = read_stl(mesh_path) * float(scale_to_mm)
    side = lumen_side_cached(tris, mode, cache)
    asset, report = import_triangles(tris, side, voxel_mm=voxel_mm, root_mm=root_mm, sites=sites,
                                     min_branch_mm=min_branch_mm, step_mm=step_mm)
    report = {"mesh": str(mesh_path), "scale_to_mm": scale_to_mm, "mode": mode, **report}
    return asset, report


def import_triangles(tris: np.ndarray, side: np.ndarray, *, voxel_mm: float = 0.5, root_mm=None,
                     sites: dict | None = None, min_branch_mm: float = 4.0, step_mm: float = 0.5,
                     provenance: str = PROVENANCE) -> tuple[Asset, dict]:
    """Convert lumen-bounding triangles (mm) into ``(asset, report)``.

    ``side`` is the per-face lumen sign from ``lumen_side`` (all -1 for a closed surface with
    outward normals). ``root_mm`` is the device inlet point; ``sites`` maps node names to points
    at which edges are split (e.g. navigation targets)."""
    t0 = time.time()
    grid, vox = voxelize(tris, side, voxel_mm)
    if root_mm is None:
        root_mm = grid.centers(np.argwhere(grid.occ)[0])
    sk = centerline_tree(grid, np.asarray(root_mm, float), min_branch_mm=min_branch_mm)
    rays = Rays(tris, side)

    def in_lumen(q):
        k = np.round((q - grid.origin) / grid.h).astype(int)
        ok = np.all((k >= 0) & (k < np.asarray(grid.occ.shape)), axis=1)
        out = np.zeros(len(q), bool)
        out[ok] = grid.occ[tuple(k[ok].T)]
        return out

    edges = []
    for a, b, poly in sk.edges:
        p = resample(poly, step_mm)
        p = smooth(p, 2.0 / step_mm)
        p = recentre(rays, p)
        p = smooth(p, 1.0 / step_mm)
        if b.startswith("end"):                 # a closed vessel end: never past the end cap
            inside = in_lumen(p)
            last = int(np.flatnonzero(inside).max()) if inside.any() else len(p) - 1
            p = p[: max(last + 1, 2)]
            sk.nodes[b] = p[-1].copy()
        edges.append((a, b, p))
    # keep the node positions consistent with the re-centred edge ends
    nodes = dict(sk.nodes)
    site_dist = {}
    for nm, xyz in (sites or {}).items():
        site_dist[nm] = _split_at(edges, nodes, nm, np.asarray(xyz, float))
    out_edges, n_clipped, n_rays = [], 0, 0
    for k, (a, b, p) in enumerate(edges):
        p = p.copy()
        p[0], p[-1] = nodes[a], nodes[b]
        th, R, _ = cross_section(rays, p)
        R, esc = fill_ostia(R)
        n_clipped += int(esc.sum())
        n_rays += R.size
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        out_edges.append(Edge(id=f"e{k}", node_a=a, node_b=b, centerline_mm=p.tolist(), s_grid=s.tolist(),
                              theta_grid=th.tolist(), R=R.tolist()))
    asset = Asset(frame=Frame(name="world_mm", spacing_mm=(1.0, 1.0, 1.0), origin_mm=(0.0, 0.0, 0.0)),
                  nodes=[Node(nm, tuple(map(float, xyz))) for nm, xyz in nodes.items()],
                  edges=out_edges, device_spawn=DeviceSpawn(node_id="inlet"), provenance=provenance)
    report = {"voxel_mm": voxel_mm,
              "step_mm": step_mm, "n_theta": N_THETA, "faces": int(len(tris)),
              "lumen_faces": int((side != 0).sum()), "voxelization": vox, "skeleton": sk.info,
              "edges": len(out_edges), "nodes": len(nodes), "site_to_centerline_mm": site_dist,
              "rays_filled_at_ostia_frac": n_clipped / max(n_rays, 1), "seconds": round(time.time() - t0, 1)}
    return asset, report


# ---------------------------------------------------------------------------------------------
# conversion error
# ---------------------------------------------------------------------------------------------
def _sample_faces(tris: np.ndarray, n: int, rng: np.random.Generator):
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    f = rng.choice(len(tris), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    T = tris[f]
    return T[:, 0] + u[:, None] * (T[:, 1] - T[:, 0]) + v[:, None] * (T[:, 2] - T[:, 0])


def _stats(d: np.ndarray) -> dict:
    d = np.abs(np.asarray(d, float))
    if len(d) == 0:
        return {"n": 0}
    return {"n": int(len(d)), "mean": float(d.mean()), "rms": float(np.sqrt((d**2).mean())),
            "p50": float(np.percentile(d, 50)), "p95": float(np.percentile(d, 95)),
            "p99": float(np.percentile(d, 99)), "max": float(d.max())}


def contact_project(tree, Q: np.ndarray, chunk: int = 256) -> dict:
    """The Newton tree-contact kernel's projection (``tube_barrier_kernel``), in NumPy.

    On every edge a point's foot is the nearest point of that edge's centerline; the point
    belongs to the edge where it has the most room there (3D distance to the foot minus the
    branch-blended radius R(s, theta) at the foot). A point beyond an OPEN vessel end gets no
    contact (``culled``), junction ends are kept; r is the radial distance at the foot and R the
    blended radius there. This is the rule that produces wall forces in simulation, so wall
    error is measured under it.
    """
    A, B, T0, T1, CS, slices = [], [], [], [], [], []
    for e in tree.edges:
        f = e.frame
        slices.append((sum(len(x) for x in A), sum(len(x) for x in A) + len(f.points) - 1))
        A.append(f.points[:-1]); B.append(f.points[1:])
        T0.append(f.tangents[:-1]); T1.append(f.tangents[1:]); CS.append(f.cum_s[:-1])
    A, B, T0, T1, CS = (np.concatenate(x) for x in (A, B, T0, T1, CS))
    AB = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-18)
    bound = np.asarray([tree._R_bound[i] for i in range(len(tree.edges))])
    out = {k: [] for k in ("edge", "s", "r", "R", "culled")}
    for c0 in range(0, len(Q), chunk):
        q = Q[c0:c0 + chunk]
        rows = np.arange(len(q))
        u = np.clip((q @ AB.T - np.einsum("ij,ij->i", A, AB)[None]) / L2[None], 0.0, 1.0)
        foot = A[None] + u[:, :, None] * AB[None]
        d2 = np.einsum("nmj,nmj->nm", q[:, None] - foot, q[:, None] - foot)
        jn = np.empty((len(q), len(slices)), int)
        dn = np.empty((len(q), len(slices)))
        for i, (s0, s1) in enumerate(slices):
            k = np.argmin(d2[:, s0:s1], axis=1)
            jn[:, i] = s0 + k
            dn[:, i] = np.sqrt(d2[rows, s0 + k])
        lower = dn - bound[None]                 # no edge can give more room than this
        order = np.argsort(lower, axis=1)
        ej = np.zeros(len(q), int)
        r, s, Rb = np.empty(len(q)), np.empty(len(q)), np.empty(len(q))
        culled = np.zeros(len(q), bool)
        for k in range(len(q)):
            best = np.inf
            for i in order[k]:
                if lower[k, i] >= best:
                    break
                j = jn[k, i]
                ft = foot[k, j]
                tang = T0[j] + u[k, j] * (T1[j] - T0[j])
                tang = tang / np.linalg.norm(tang)
                rad = (q[k] - ft) - np.dot(q[k] - ft, tang) * tang
                sk = CS[j] + u[k, j] * np.sqrt(L2[j])
                theta = tree.edges[i].frame.project(ft + rad).theta   # frame-consistent theta
                R = tree.blended_R(int(i), float(sk), float(theta))
                if dn[k, i] - R < best:
                    best = dn[k, i] - R
                    ej[k], r[k], s[k], Rb[k] = i, np.linalg.norm(rad), sk, R
            e = tree.edges[ej[k]]
            f = e.frame
            if not tree.is_junction(e.node_a) and np.dot(q[k] - f.points[0], f.tangents[0]) < 0:
                culled[k] = True
            if not tree.is_junction(e.node_b) and np.dot(q[k] - f.points[-1], f.tangents[-1]) > 0:
                culled[k] = True
        for key, v in (("edge", ej), ("s", s), ("r", r), ("R", Rb), ("culled", culled)):
            out[key].append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def conversion_error(asset: Asset, mesh_path: str | Path | None = None, *, scale_to_mm: float = 1000.0, mode: str = "shell",
                     n_samples: int = 300_000, seed: int = 0, task_edges: set | None = None,
                     tris: np.ndarray | None = None, side: np.ndarray | None = None) -> dict:
    """Bidirectional wall distance (mm) between the source lumen surface and the Lumen wall.

    source -> Lumen: lumen-surface samples projected onto the tree, |r - R(s, theta)| (branch-blended R).
    Lumen -> source: Lumen wall points c(s) + R e(theta) to the nearest source surface sample, excluding
    points that lie inside another edge's lumen (virtual wall across an ostium).
    ``task_edges`` restricts a second set of statistics to the edges a navigation task uses.
    """
    from scipy.spatial import cKDTree

    from lumen.core.tree import VascularTree

    rng = np.random.default_rng(seed)
    if tris is None:
        tris = read_stl(mesh_path) * float(scale_to_mm)
    if side is None:
        side = lumen_side(tris, mode)
    S = _sample_faces(tris[side != 0], n_samples, rng)
    tree = VascularTree(asset)
    Ssub = S[:: max(1, n_samples // 60_000)]
    cp = contact_project(tree, Ssub)
    keep = ~cp["culled"]                      # open-end caps: no contact by design in Lumen
    d_src_all = cp["r"] - cp["R"]
    d_src, e_src = d_src_all[keep], cp["edge"][keep]
    d_rule = []                               # the python diagnostic rule (tree.project, min r)
    for q in Ssub[:: 6]:
        pr = tree.project(q)
        d_rule.append(pr.r - pr.R)
    d_rule = np.asarray(d_rule)
    # junction zones: within the widest meeting vessel's radius of a branch point
    node_pos = {n.id: np.asarray(n.position_mm, float) for n in asset.nodes}
    deg: dict[str, list] = {}
    for e in asset.edges:
        R = np.asarray(e.R)
        deg.setdefault(e.node_a, []).append(float(np.median(R[0])))
        deg.setdefault(e.node_b, []).append(float(np.median(R[-1])))
    jn = [(node_pos[k], max(v)) for k, v in deg.items() if len(v) >= 3]
    in_j = np.zeros(len(Ssub), bool)
    for c, r in jn:
        in_j |= np.linalg.norm(Ssub - c, axis=1) <= r
    in_j = in_j[keep]
    kd = cKDTree(S)
    d_lum, e_lum = [], []
    for ei, e in enumerate(asset.edges):
        pts, lf = asset.edge_arrays(e)
        _, t, m1, m2 = _normal_frame(pts)
        th = np.asarray(e.theta_grid)
        W = pts[:, None, :] + np.asarray(e.R)[:, :, None] * (np.cos(th)[None, :, None] * m1[:, None, :]
                                                             + np.sin(th)[None, :, None] * m2[:, None, :])
        W = W.reshape(-1, 3)
        cw = contact_project(tree, W)
        real = ~((cw["edge"] != ei) & (cw["r"] < 0.9 * cw["R"]))
        d, _ = kd.query(W[real])
        d_lum.append(d)
        e_lum.append(np.full(len(d), ei))
    d_lum, e_lum = np.concatenate(d_lum), np.concatenate(e_lum)
    out = {"source_to_lumen_mm": _stats(d_src), "lumen_to_source_mm": _stats(d_lum),
           "source_to_lumen_signed_mean_mm": float(d_src.mean()) if len(d_src) else None,
           "source_samples_at_open_end_caps_frac": float(cp["culled"].mean()),
           "source_to_lumen_tree_project_rule_mm": _stats(d_rule),
           "source_to_lumen_outside_junction_zones_mm": _stats(d_src[~in_j]),
           "source_to_lumen_junction_zones_mm": _stats(d_src[in_j]),
           "junction_zone_surface_fraction": float(in_j.mean()),
           "definition": "source_to_lumen: source lumen-surface samples, |r - R| under the Newton tree-contact "
                         "kernel's projection (each edge's nearest foot, owned by the edge with most room; branch-blended R; samples on open vessel "
                         "end caps excluded, where Lumen applies no contact); tree_project_rule: the same under "
                         "VascularTree.project (min radial distance, used for observations); lumen_to_source: Lumen wall points to the "
                         "nearest source surface sample (virtual walls across ostia excluded); junction zone: "
                         "within the widest meeting vessel's radius of a branch point"}
    if task_edges:
        sel = np.isin(e_lum, sorted(task_edges))
        out["task_edges"] = sorted(task_edges)
        out["lumen_to_source_task_mm"] = _stats(d_lum[sel])
        es = np.asarray(e_src)
        if (es >= 0).any():
            out["source_to_lumen_task_mm"] = _stats(d_src[np.isin(es, sorted(task_edges))])
    return out


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def _plots(asset: Asset, tris: np.ndarray, side: np.ndarray, sites, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = _sample_faces(tris[side != 0], 60_000, np.random.default_rng(1))
    files = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, (i, j, lab) in zip(axes, [(0, 1, "x-y (mm)"), (2, 1, "z-y (mm)")]):
        ax.scatter(S[:, i], S[:, j], s=0.05, c="0.75")
        for e in asset.edges:
            p = np.asarray(e.centerline_mm)
            ax.plot(p[:, i], p[:, j], lw=1.2)
        for n in asset.nodes:
            ax.plot(n.position_mm[i], n.position_mm[j], "k.", ms=4)
            ax.annotate(n.id, (n.position_mm[i], n.position_mm[j]), fontsize=6)
        for nm, xyz in (sites or {}).items():
            ax.plot(xyz[i], xyz[j], "r*", ms=10)
        ax.set_aspect("equal")
        ax.set_title(f"source lumen surface (grey), imported centerlines, nodes, target sites (red) {lab}", fontsize=8)
    f = out / "import_centerlines.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(str(f))
    # cross-sections on the longest edges
    big = sorted(asset.edges, key=lambda e: -len(e.centerline_mm))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, e in zip(axes.ravel(), big):
        pts = np.asarray(e.centerline_mm)
        k = len(pts) // 2
        _, t, m1, m2 = _normal_frame(pts)
        near = S[np.abs((S - pts[k]) @ t[k]) < 0.4]
        rel = near - pts[k]
        u, v = rel @ m1[k], rel @ m2[k]
        lim = 2.2 * float(np.max(e.R[k]))
        sel = (np.abs(u) < lim) & (np.abs(v) < lim)
        ax.scatter(u[sel], v[sel], s=2, c="0.6", label="source surface (slab +-0.4 mm)")
        th = np.asarray(e.theta_grid)
        R = np.asarray(e.R[k])
        ax.plot(np.r_[R * np.cos(th), R[0] * np.cos(th[0])], np.r_[R * np.sin(th), R[0] * np.sin(th[0])], "r-",
                label="Lumen R(s, theta)")
        ax.set_aspect("equal")
        ax.set_title(f"{e.id} {e.node_a}-{e.node_b} s={e.s_grid[k]:.1f} mm", fontsize=8)
    axes.ravel()[0].legend(fontsize=6)
    f = out / "import_cross_sections.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(str(f))
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True, help="output directory (asset JSON, report, figures)")
    ap.add_argument("--scale-to-mm", type=float, default=1000.0)
    ap.add_argument("--mode", choices=["shell", "solid"], default="shell")
    ap.add_argument("--voxel-mm", type=float, default=0.5)
    ap.add_argument("--cache", default=None, help="directory for the face-label cache (keeps peak memory low)")
    ap.add_argument("--labels-only", action="store_true", help="compute and cache the face labels, then exit")
    ap.add_argument("--root", default=None, help="inlet point x,y,z in mm")
    ap.add_argument("--site", action="append", default=[], help="name=x,y,z (mm); repeatable")
    ap.add_argument("--task-route", action="append", default=[], help="node names whose route from the inlet "
                    "defines the task edges for error statistics; repeatable")
    a = ap.parse_args(argv)
    if a.voxel_mm <= 0 or a.scale_to_mm <= 0:
        raise ValueError("--voxel-mm and --scale-to-mm must be positive")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sites = {}
    for s in a.site:
        nm, xyz = s.split("=", 1)
        sites[nm] = [float(v) for v in xyz.split(",")]
    root = [float(v) for v in a.root.split(",")] if a.root else None
    if a.labels_only:
        lumen_side_cached(read_stl(a.mesh) * a.scale_to_mm, a.mode, a.cache or out)
        return
    asset, rep = import_mesh(a.mesh, scale_to_mm=a.scale_to_mm, mode=a.mode, voxel_mm=a.voxel_mm,
                             root_mm=root, sites=sites, cache=a.cache)
    asset.save(str(out / "asset.json"))
    from lumen.core.tree import VascularTree
    tree = VascularTree(asset)
    ids = [n.id for n in asset.nodes]
    task_edges = set()
    rep["routes"] = {}
    for nm in a.task_route:
        r = tree.route(nm, "inlet")
        task_edges.update(r)
        rep["routes"][nm] = {"edges": [int(x) for x in r], "length_mm": float(tree.route_length(r))}
    rep["site_projection"] = {}
    for nm, xyz in sites.items():
        pr = tree.project(np.asarray(xyz))
        rep["site_projection"][nm] = {"r_mm": float(pr.r), "R_mm": float(pr.R),
                                      "inside": bool(pr.r < pr.R), "node_present": nm in ids}
    if root is not None:
        pr = tree.project(np.asarray(root))
        rep["inlet_projection"] = {"r_mm": float(pr.r), "R_mm": float(pr.R)}
    rep["radius_mm_by_edge"] = {e.id: {"nodes": [e.node_a, e.node_b], "length_mm": float(e.s_grid[-1]),
                                       "R_min": float(np.min(e.R)), "R_median": float(np.median(e.R)),
                                       "R_max": float(np.max(e.R))} for e in asset.edges}
    tris = read_stl(a.mesh) * a.scale_to_mm
    side = lumen_side_cached(tris, a.mode, a.cache)
    rep["conversion_error"] = conversion_error(asset, a.mesh, task_edges=task_edges or None, tris=tris, side=side)
    try:
        rep["figures"] = _plots(asset, tris, side, sites, out)
    except ImportError:                          # figures are optional; the report is not
        rep["figures"] = []
        print("figures skipped: matplotlib is not installed")
    (out / "import_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("voxelization", "skeleton", "edges", "routes", "site_projection",
                                          "conversion_error", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
