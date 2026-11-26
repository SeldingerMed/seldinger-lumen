"""What the viewer draws: one anatomy as a JSON scene (mesh, centrelines, names, entry).

Anatomy can be a built-in generated scene (``arch``, ``tortuous``, ``bifurcation``, ``tube``), a
case shipped with Lumen (see ``lumen.cases``), an ``lumen import`` output folder, or an asset
JSON file. Everything stays in the asset's own millimetre frame, so a recorded run (whose wire
positions are in the same frame) lines up without any registration.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lumen.assets.mesh import load_npz, simplify_to, weld
from lumen.assets.schema import Asset

SCENE_VERSION = "lumen-scene/0"
MAX_PAGE_TRIANGLES = 250_000      # keeps the page a few MB and smooth on a laptop

BUILT_IN = {
    "arch": ("aortic_arch_tree", "Generated aortic arch with its three great vessels"),
    "tortuous": ("tortuous_tree", "Generated tortuous branching tree"),
    "bifurcation": ("bifurcation", "Generated single bifurcation"),
    "tube": ("straight_tube", "Generated straight tube"),
}


@dataclass
class Anatomy:
    """An anatomy ready to draw and to drive: the asset plus what an import knows about it."""

    asset: Asset
    label: str
    names: dict = field(default_factory=lambda: {"edges": {}, "vessels": {}})
    surface: tuple | None = None               # (vertices, faces) of the scan's vessel surface
    folder: Path | None = None                 # an import folder, when it came from one
    source: dict = field(default_factory=dict)  # where it came from (and its licence)


def resolve(spec: str | Path) -> Anatomy:
    """Turn a name or a path into an :class:`Anatomy`."""
    from lumen import cases
    from lumen.assets import procedural

    text = str(spec)
    if text in BUILT_IN:
        fn, label = BUILT_IN[text]
        asset = getattr(procedural, fn)()
        return Anatomy(asset, label, _leaf_names(asset), source={"kind": "generated", "name": text})
    folder = cases.case_folder(text)
    if folder is None:
        p = Path(text).expanduser()
        if p.is_dir() and (p / "asset.json").exists():
            folder = p
        elif p.is_file() and p.suffix == ".json":
            return Anatomy(Asset.load(str(p)), p.stem, source={"kind": "asset", "path": str(p)})
        else:
            known = ", ".join(list(BUILT_IN) + cases.names())
            raise ValueError(f"no anatomy called {text!r}: give an import folder, an asset .json "
                             f"file, or one of: {known}")
    from lumen.scan.pipeline import load_imported

    asset, names = load_imported(folder)
    surface = None
    stl, npz = folder / "surface.stl", folder / "surface.npz"
    if npz.exists():
        surface = load_npz(npz)
    elif stl.exists():
        from lumen.assets.surface_import import read_stl

        surface = weld(np.asarray(read_stl(stl), np.float32))
    source = {"kind": "import", "path": str(folder)}
    card = folder / "case.json"
    if card.exists():
        try:
            source = {"kind": "case", **json.loads(card.read_text())}
        except ValueError:
            pass
    label = source.get("title") or folder.name
    return Anatomy(asset, label, names, surface, folder, source)


def _leaf_names(asset: Asset) -> dict:
    """Generated anatomy has no vessel names; call each end after its node (``carotid_out`` ->
    ``carotid``) so it can be labelled and aimed at like an imported one."""
    degree: dict = {}
    for e in asset.edges:
        for n in (e.node_a, e.node_b):
            degree[n] = degree.get(n, 0) + 1
    start = asset.device_spawn.node_id
    vessels = {}
    for n in asset.nodes:
        if degree.get(n.id, 0) == 1 and n.id != start:
            name = n.id[:-4] if n.id.endswith("_out") else n.id
            vessels[name.replace("_", " ")] = {"target_node": n.id}
    return {"edges": {}, "vessels": vessels}


# ---------------------------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------------------------
def tube_mesh(asset: Asset, n_theta: int = 20, step_mm: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Sweep every edge's lumen R(s, theta) into a closed-sided tube: ``(vertices, faces)``.

    This is the wall the solver sees. Generated anatomy has nothing else; imported anatomy also
    has the scan's own surface, which looks better and is drawn instead when present.
    """
    from lumen.core.frame import CenterlineFrame

    verts, faces, base = [], [], 0
    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    for e in asset.edges:
        pts, lf = asset.edge_arrays(e)
        fr = CenterlineFrame(np.asarray(pts, float))
        n = max(2, int(np.ceil(fr.length / step_mm)) + 1)
        s = np.linspace(0.0, fr.length, n)
        idx = np.clip(np.searchsorted(fr.cum_s, s, side="right") - 1, 0, len(fr.points) - 2)
        u = (s - fr.cum_s[idx]) / np.maximum(fr.cum_s[idx + 1] - fr.cum_s[idx], 1e-9)
        c = fr.points[idx] + u[:, None] * (fr.points[idx + 1] - fr.points[idx])
        m1 = fr.m1[idx]
        m2 = fr.m2[idx]
        sg = s * (lf.s[-1] / fr.length) if fr.length > 0 else s
        R = np.array([[lf.eval(float(a), float(t)) for t in th] for a in sg])
        ring = (c[:, None, :] + R[:, :, None] * (np.cos(th)[None, :, None] * m1[:, None, :]
                                                 + np.sin(th)[None, :, None] * m2[:, None, :]))
        i = np.arange(n - 1)[:, None]
        j = np.arange(n_theta)[None, :]
        a = base + i * n_theta + j
        b = base + i * n_theta + (j + 1) % n_theta
        side = np.stack([a, b, a + n_theta, b, b + n_theta, a + n_theta], -1).reshape(-1, 3)
        # closed ends: the C-arm view measures the path through each closed surface
        c0, c1 = base + n * n_theta, base + n * n_theta + 1
        jj = np.arange(n_theta)
        last = base + (n - 1) * n_theta
        caps = np.concatenate([np.stack([np.full(n_theta, c0), base + (jj + 1) % n_theta, base + jj], 1),
                               np.stack([np.full(n_theta, c1), last + jj, last + (jj + 1) % n_theta], 1)])
        tri = np.concatenate([side, caps])
        pts = np.concatenate([ring.reshape(-1, 3), c[:1], c[-1:]])
        # face outward whatever handedness the frame has
        p0, p1, p2 = (pts[tri[0, k] - base] for k in range(3))
        if np.dot(np.cross(p1 - p0, p2 - p0), p0 - c[0]) < 0:
            tri = tri[:, ::-1]
        verts.append(pts)
        faces.append(tri)
        base += len(pts)
    return (np.concatenate(verts).astype(np.float32), np.concatenate(faces).astype(np.uint32))


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _round(a, nd=2):
    return np.round(np.asarray(a, float), nd).tolist()


def scene_dict(anatomy: Anatomy, target: str | None = None) -> dict:
    """The JSON the page draws."""
    asset = anatomy.asset
    if anatomy.surface is not None and len(anatomy.surface[1]):
        verts, faces = simplify_to(*anatomy.surface, max_triangles=MAX_PAGE_TRIANGLES)
        mesh_kind = "scan surface"
    else:
        verts, faces = tube_mesh(asset)
        mesh_kind = "lumen model"
    nodes = {n.id: np.asarray(n.position_mm, float) for n in asset.nodes}
    labels = []
    for name, row in sorted((anatomy.names or {}).get("vessels", {}).items()):
        node = row.get("target_node")
        if node in nodes:
            labels.append({"name": name, "at": _round(nodes[node], 1), "node": node})
    lines = []
    for e in asset.edges:
        pts = np.asarray(e.centerline_mm, float)
        keep = np.unique(np.r_[np.arange(0, len(pts), 3), len(pts) - 1])
        lines.append({"id": e.id, "name": (anatomy.names or {}).get("edges", {}).get(e.id, ""),
                      "points": _round(pts[keep], 2)})
    lo, hi = verts.min(0), verts.max(0)
    return {
        "version": SCENE_VERSION,
        "label": anatomy.label,
        "frame": asset.frame.name,
        "mesh": {"kind": mesh_kind, "positions": _b64(verts), "indices": _b64(faces),
                 "vertex_count": int(len(verts)), "triangle_count": int(len(faces))},
        "centerlines": lines,
        "labels": labels,
        "entry": _round(nodes[asset.device_spawn.node_id], 2),
        "target": target,
        "bounds": {"min": _round(lo, 2), "max": _round(hi, 2)},
        "source": anatomy.source,
    }
