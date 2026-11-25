"""Real-scan anatomy: a labelled vessel mask on a scan grid -> a navigable Lumen ``Asset``.

``lumen import`` segments a CTA or MRA (or takes a mask the user already has) and calls
``asset_from_vessel_mask``. The mask is cleaned, turned into a closed surface in patient
millimetres, and converted by ``lumen.assets.surface_import`` into a centerline tree with a
fitted wall on every edge. The report states how far the fitted wall is from the mask surface.

Coordinates: everything is written in the DICOM patient frame, LPS, in millimetres (x toward the
patient's left, y toward the back, z toward the head). NIfTI (RAS) input is converted on read.

Needs the ``scan`` extra (SciPy, scikit-image); readers need pynrrd / nibabel / pydicom only for
the formats actually opened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import json

import numpy as np

from lumen.assets.schema import Asset, Frame

FRAME_NAME = "patient_lps_mm"
PROVENANCE = "imported(scan)"


@dataclass
class LabelVolume:
    """An integer label volume with its voxel-index -> patient LPS (mm) affine."""

    data: np.ndarray            # (nx, ny, nz) integer labels, 0 = background
    affine: np.ndarray          # 4x4, maps (i, j, k, 1) to LPS mm
    names: dict                 # label value -> vessel name ("" if unnamed)

    @property
    def spacing_mm(self) -> np.ndarray:
        return np.linalg.norm(self.affine[:3, :3], axis=0)


_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def save_label_volume(volume: LabelVolume, path: str | Path) -> None:
    """Write a label volume as NIfTI, with the vessel names in the description header."""
    import json

    import nibabel as nib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(volume.data, dtype=np.int16),
                          _RAS_TO_LPS @ np.asarray(volume.affine, float))
    img.header["descrip"] = json.dumps({str(k): v for k, v in volume.names.items()})[:79].encode()
    img.header.set_xyzt_units("mm")
    nib.save(img, str(p))
    sidecar = p.with_suffix("").with_suffix(".names.json") if p.name.endswith(".nii.gz")         else p.with_suffix(".names.json")
    sidecar.write_text(json.dumps({str(k): v for k, v in volume.names.items()}))


def load_label_volume(path: str | Path, names: dict | None = None) -> LabelVolume:
    """Read a mask from NRRD (``.nrrd``, ``.seg.nrrd``) or NIfTI (``.nii``, ``.nii.gz``)."""
    p = Path(path)
    low = p.name.lower()
    if low.endswith(".nrrd"):
        import nrrd
        data, hdr = nrrd.read(str(p), index_order="F")
        if data.ndim != 3:
            raise ValueError(f"expected a 3-D label volume, got shape {data.shape}: {p}")
        dirs = np.asarray(hdr["space directions"], float)
        if dirs.shape != (3, 3) or not np.isfinite(dirs).all():
            # a 3-D volume needs three spatial axes; a 'none' (NaN) row means one is missing
            raise ValueError(f"the NRRD 'space directions' header does not give three spatial axes "
                             f"({dirs.tolist()}): {p}")
        aff = np.eye(4)
        aff[:3, :3] = dirs.T
        aff[:3, 3] = np.asarray(hdr["space origin"], float)
        space = str(hdr.get("space", "left-posterior-superior")).lower()
        if space in ("right-anterior-superior", "ras"):
            aff = _RAS_TO_LPS @ aff
        elif space not in ("left-posterior-superior", "lps"):
            raise ValueError(f"unsupported NRRD space {space!r}: {p}")
        found = {}
        for k, v in hdr.items():
            if k.startswith("Segment") and k.endswith("_LabelValue"):
                found[int(v)] = str(hdr.get(k.replace("_LabelValue", "_Name"), ""))
        return LabelVolume(np.asarray(data), aff, {**found, **(names or {})})
    if low.endswith(".nii") or low.endswith(".nii.gz"):
        import nibabel as nib
        img = nib.load(str(p))
        data = np.asarray(img.dataobj)
        if data.ndim == 4 and data.shape[3] == 1:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"expected a 3-D label volume, got shape {data.shape}: {p}")
        sidecar = p.with_suffix("").with_suffix(".names.json") if p.name.endswith(".nii.gz")             else p.with_suffix(".names.json")
        saved = {}
        if sidecar.exists():
            saved = {int(k): v for k, v in json.loads(sidecar.read_text()).items()}
        return LabelVolume(np.rint(data).astype(np.int32), _RAS_TO_LPS @ np.asarray(img.affine, float),
                           {**saved, **(names or {})})
    raise ValueError(f"unsupported mask format (use .nrrd, .nii or .nii.gz): {p}")


JOIN_MM = 2.0            # pieces closer than this to the main vessel are joined to it
JOIN_MIN_MM3 = 20.0      # ... unless they are specks smaller than this


def _join_near_pieces(vol, lab, keep, h, join_mm, min_mm3):
    """Join pieces that sit within ``join_mm`` of the kept piece with a thin straight bridge.

    A segmented vessel is often broken by a gap of a voxel or two where it is thin. Such a
    piece is the same vessel, so it is welded on at the closest point. ``vol`` and ``lab`` are
    changed in place; returns one record per join.
    """
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    def shell(mask_box):
        return mask_box & ~ndi.binary_erosion(mask_box)

    joined = []
    radius = max(0.6, 1.5 * h)
    boxes = ndi.find_objects(lab)
    sizes = np.bincount(lab.ravel())
    pending = [i for i in range(1, len(sizes)) if i != keep and sizes[i] * h**3 >= min_mm3]
    main_pts = np.argwhere(shell(lab == keep))
    while pending:
        tree = cKDTree(main_pts * h)
        best = None
        for i in pending:
            box = boxes[i - 1]
            pts = np.argwhere(shell(lab[box] == i)) + [s.start for s in box]
            d, j = tree.query(pts * h)
            k = int(np.argmin(d))
            if d[k] <= join_mm and (best is None or d[k] < best[0]):
                best = (float(d[k]), i, pts[k], main_pts[int(j[k])], pts)
        if best is None:
            break
        gap, i, a, b, pts = best
        lo = np.maximum(np.minimum(a, b) - int(np.ceil(radius / h)) - 2, 0)
        hi = np.minimum(np.maximum(a, b) + int(np.ceil(radius / h)) + 3, vol.shape)
        g = np.stack(np.meshgrid(*[np.arange(x, y) for x, y in zip(lo, hi)], indexing="ij"), -1) * h
        pa, pb = a * h, b * h
        ab = pb - pa
        u = np.clip(((g - pa) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
        dist = np.linalg.norm(g - (pa + u[..., None] * ab), axis=-1)
        box = tuple(slice(x, y) for x, y in zip(lo, hi))
        vol[box] = np.maximum(vol[box], np.clip(0.5 + (radius - dist) / h, 0.0, 1.0))
        fill = (dist <= radius) & (lab[box] == 0)
        bridge = np.argwhere(fill) + lo
        lab[box][fill] = keep
        lab[boxes[i - 1]][lab[boxes[i - 1]] == i] = keep
        main_pts = np.vstack([main_pts, pts, bridge])
        pending.remove(i)
        joined.append({"gap_mm": round(gap, 2), "volume_mm3": round(float(sizes[i] * h**3), 1),
                       "at_ijk": [float(v) for v in 0.5 * (a + b)]})
    return joined


def surface_from_mask(mask: np.ndarray, affine: np.ndarray, *, iso_mm: float | None = None,
                      smooth_mm: float | None = None, join_mm: float = JOIN_MM) -> tuple[np.ndarray, dict]:
    """Closed triangle surface (F, 3, 3) in LPS mm with outward normals from a binary mask.

    The mask is cropped, resampled to an isotropic grid (so thick slices do not leave stair
    steps), lightly smoothed and contoured at one half. Pieces within ``join_mm`` of the largest
    piece are joined to it; anything farther away is dropped, because a wire cannot cross a
    gap. The grid is zero padded, so a vessel that leaves the scan is capped flat at the scan
    boundary.
    """
    from scipy import ndimage as ndi
    from skimage import measure

    A = np.asarray(affine, float)
    sp = np.linalg.norm(A[:3, :3], axis=0)
    if not np.all(sp > 0):
        raise ValueError("the affine has a zero-length axis")
    Q = A[:3, :3] / sp                                  # direction cosines (columns)
    if not np.allclose(Q.T @ Q, np.eye(3), atol=1e-3):
        raise ValueError("the scan axes are not perpendicular (gantry tilt is not supported)")
    idx = np.argwhere(mask)
    if len(idx) == 0:
        raise ValueError("the vessel mask is empty")
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = np.minimum(idx.max(0) + 3, mask.shape)
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    h = float(iso_mm) if iso_mm else float(np.clip(sp.min(), 0.4, 1.0))
    vol = ndi.zoom(sub, sp / h, order=1, mode="grid-constant", cval=0.0, grid_mode=True)
    # smooth each axis by its own slice spacing: enough to remove the stair steps of thick
    # slices without shrinking thin vessels in the finely sampled directions
    sig = (np.full(3, float(smooth_mm)) if smooth_mm is not None else 0.6 * sp) / h
    sig = np.maximum(sig, 0.5)
    vol = ndi.gaussian_filter(vol, sig, mode="constant")
    vol = np.pad(vol, 2)
    lab, n = ndi.label(vol > 0.5)                       # face connectivity, as the converter uses
    if n == 0:
        raise ValueError("the vessel is too thin to survive smoothing; lower smooth_mm")
    sizes = np.bincount(lab.ravel())[1:]
    keep = int(np.argmax(sizes)) + 1
    joined = []
    if n > 1:
        if join_mm > 0:
            joined = _join_near_pieces(vol, lab, keep, h, float(join_mm), JOIN_MIN_MM3)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        # blank the remaining pieces and their halos
        other = ndi.binary_dilation((lab > 0) & (lab != keep), iterations=2)
        vol = np.where(other & (lab != keep), 0.0, vol)
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=(h, h, h))
    # grid_mode zoom: output voxel n has its centre at (n + 0.5) * h - 0.5 * spacing, in mm along
    # each scan axis from the centre of input voxel `lo`; undo the padding too
    mm = verts - 2 * h + 0.5 * h - 0.5 * sp
    world = (Q @ mm.T).T + (A @ np.r_[lo, 1.0])[:3]
    tris = world[faces]
    c = tris.mean(1)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    vol6 = float(np.einsum("ij,ij->i", c - c.mean(0), nrm).sum())   # 6 x signed volume
    if vol6 < 0:                                        # make the normals point out of the vessel
        tris = tris[:, ::-1]
    dropped = np.sort(np.delete(sizes, keep - 1))[::-1]
    dropped = dropped[dropped > 0]
    for j in joined:                                    # report where each join is, in patient mm
        at = np.asarray(j.pop("at_ijk")) * h - 2 * h + 0.5 * h - 0.5 * sp
        j["at_mm"] = [round(float(v), 1) for v in Q @ at + (A @ np.r_[lo, 1.0])[:3]]
    info = {"iso_mm": h, "smooth_sigma_mm": [float(v * h) for v in sig], "faces": int(len(tris)),
            "surface_volume_mm3": abs(vol6) / 6.0,
            "mask_volume_mm3": float(mask.sum() * np.prod(sp)),
            "scan_spacing_mm": [float(v) for v in sp],
            "pieces": int(n), "joined": joined, "dropped_volume_mm3": float(dropped.sum() * h**3),
            "largest_dropped_piece_mm3": float(dropped[0] * h**3) if len(dropped) else 0.0}
    return tris, info


def default_entry(tris: np.ndarray, slab_mm: float = 3.0) -> np.ndarray:
    """The device entry when the user gives none: the most caudal end of the vessel (a wire
    comes up from the groin, or from the bottom of a neck or head scan)."""
    v = tris.reshape(-1, 3)
    low = v[np.argmin(v[:, 2])]
    near = v[(v[:, 2] <= low[2] + slab_mm) & (np.linalg.norm(v - low, axis=1) <= 6 * slab_mm)]
    c = near.mean(0)
    c[2] = low[2] + 0.5 * slab_mm                       # just inside the end cap
    return c


def asset_from_vessel_mask(volume: LabelVolume, *, entry_mm=None, voxel_mm: float | None = None,
                           min_branch_mm: float = 6.0, step_mm: float = 0.5,
                           error_samples: int = 120_000) -> tuple[Asset, dict, np.ndarray]:
    """Convert a vessel mask into ``(asset, report, surface_triangles)``.

    Every non-zero label counts as vessel. ``entry_mm`` is the device entry in LPS mm (default:
    the most caudal point of the vessel).
    """
    from lumen.assets import surface_import as si

    t0 = time.time()
    tris, surf = surface_from_mask(np.asarray(volume.data) != 0, volume.affine)
    side = -np.ones(len(tris))
    root = np.asarray(entry_mm, float) if entry_mm is not None else default_entry(tris)
    h = float(voxel_mm) if voxel_mm else float(np.clip(surf["iso_mm"], 0.5, 1.0))
    asset, rep = si.import_triangles(tris, side, voxel_mm=h, root_mm=root, min_branch_mm=min_branch_mm,
                                     step_mm=step_mm, provenance=PROVENANCE)
    asset.frame = Frame(name=FRAME_NAME, spacing_mm=(1.0, 1.0, 1.0), origin_mm=(0.0, 0.0, 0.0))
    err = si.conversion_error(asset, tris=tris, side=side, n_samples=error_samples)
    radii = [float(np.median(np.asarray(e.R))) for e in asset.edges]
    lengths = [float(e.s_grid[-1]) for e in asset.edges]
    report = {"frame": FRAME_NAME, "entry_mm": [float(v) for v in root], "surface": surf,
              "conversion": rep, "wall_error_mm": err,
              "edges": [{"id": e.id, "from": e.node_a, "to": e.node_b, "length_mm": round(L, 2),
                         "median_radius_mm": round(r, 2)}
                        for e, L, r in zip(asset.edges, lengths, radii)],
              "seconds": round(time.time() - t0, 1)}
    return asset, report, tris
