"""Sharpen a segmenter's artery labels against the scan they came from.

Segmentation models work on a coarse grid (TotalSegmentator at 1.5 mm) and tend to draw a vessel
slightly fat: the wall and a rim of partial-volume voxels are included. For navigation the
surface that matters is the contrast-filled lumen. In a CTA or a bright-blood MRA that lumen is
visible in the picture itself, so each labelled vessel is redrawn at the half-way brightness
between its own lumen and the tissue around it, staying within a thin band of the model's
outline so that nearby bone or veins cannot leak in.

A scan with no bright lumen (no contrast) is left as the model drew it.

Models also miss short stretches of a vessel, which leaves its far part cut off from the rest.
Where the scan shows contrast running across such a gap, the missing stretch is filled in.
"""

from __future__ import annotations

import numpy as np

from lumen.scan.io import is_ct

BAND_MM = 2.0            # how far the refined wall may move from the model's outline
MIN_CONTRAST = 60.0      # CT: lumen must be this many HU brighter than its surroundings
BRIDGE_MM = 15.0         # longest missed stretch that is filled in along the scan's contrast
BRIDGE_SAME_MM = 30.0    # ... when both ends carry the same vessel name
BRIDGE_MIN_MM3 = 50.0    # pieces smaller than this are not worth reconnecting


def _ball(radius_mm: float, spacing) -> np.ndarray:
    r = [max(1, int(np.ceil(radius_mm / s))) for s in spacing]
    g = np.meshgrid(*[np.arange(-n, n + 1) * s for n, s in zip(r, spacing)], indexing="ij")
    return (g[0] ** 2 + g[1] ** 2 + g[2] ** 2) <= radius_mm ** 2 + 1e-9


def refine_labels(scan, labels, band_mm: float = BAND_MM) -> tuple[np.ndarray, dict]:
    """Return ``(refined label array, report)``; the input label volume is not modified.

    The report has ``vessels`` (what was done to each named vessel) and ``bridged`` (missed
    stretches that were filled in).

    ``scan`` is a ``ScanVolume`` and ``labels`` a ``LabelVolume`` on the same grid.
    """
    from scipy import ndimage as ndi

    img = np.asarray(scan.data)
    lab = np.asarray(labels.data)
    if img.shape != lab.shape:
        raise ValueError("the scan and its labels are on different grids")
    sp = np.linalg.norm(np.asarray(labels.affine, float)[:3, :3], axis=0)
    out = np.zeros(lab.shape, lab.dtype)
    report: dict = {}
    levels: dict = {}                    # label value -> (lumen/tissue half-way level, upper cap)
    ct = is_ct(scan)
    ball_band, ball_core = _ball(band_mm, sp), _ball(min(1.5, band_mm), sp)
    for value in [int(v) for v in np.unique(lab) if v > 0]:
        name = labels.names.get(value, str(value))
        whole = lab == value
        idx = np.nonzero(whole)
        pad = [int(np.ceil((band_mm + 4.0) / s)) + 1 for s in sp]
        box = tuple(slice(max(int(i.min()) - p, 0), int(i.max()) + p + 1) for i, p in zip(idx, pad))
        m, im = whole[box], img[box]
        core = ndi.binary_erosion(m, ball_core)
        if core.sum() < 30:
            core = m                     # a thin vessel: no interior to erode to
        band = ndi.binary_dilation(m, ball_band)
        far = ndi.binary_dilation(band, ball_band) & ~band & (lab[box] == 0)
        if far.sum() < 30:
            out[box][m] = value
            report[name] = {"refined": False, "why": "no surrounding tissue to compare with"}
            continue
        lumen, around = float(np.median(im[core])), float(np.median(im[far]))
        need = MIN_CONTRAST if ct else 0.25 * abs(lumen)
        if lumen - around < need:
            out[box][m] = value
            report[name] = {"refined": False, "why": "the lumen is not brighter than its surroundings",
                            "lumen": round(lumen, 1), "around": round(around, 1)}
            continue
        level = 0.5 * (lumen + around)
        bright = (im >= level) & band & ((lab[box] == 0) | m)
        # keep only what is attached to the model's own vessel
        comp, n = ndi.label(bright)
        if n == 0:
            out[box][m] = value
            report[name] = {"refined": False, "why": "nothing bright inside the outline"}
            continue
        seed = np.unique(comp[core & bright])
        keep = np.isin(comp, seed[seed > 0])
        if keep.sum() < 0.3 * m.sum():       # the picture disagrees with the model: trust the model
            out[box][m] = value
            report[name] = {"refined": False, "why": "the bright region is much smaller than the outline",
                            "lumen": round(lumen, 1), "around": round(around, 1)}
            continue
        sub = out[box]
        sub[keep & (sub == 0)] = value
        report[name] = {"refined": True, "lumen": round(lumen, 1), "around": round(around, 1),
                        "level": round(level, 1),
                        "volume_ratio": round(float(keep.sum()) / float(m.sum()), 3)}
        levels[value] = (level, lumen + (lumen - around))
    affine = np.asarray(labels.affine, float)
    bridged = []
    # a named vessel should be one piece: first mend each vessel, then the tree as a whole
    pad = np.ceil((BRIDGE_SAME_MM + 3.0) / sp).astype(int)
    for value, box in enumerate(ndi.find_objects(out), 1):
        if box is None or value not in levels:
            continue
        box = tuple(slice(max(s.start - p, 0), s.stop + p) for s, p in zip(box, pad))
        comp, n = ndi.label(out[box] == value, structure=np.ones((3, 3, 3)))
        shift = np.eye(4)
        shift[:3, 3] = [s.start for s in box]
        bridged += _bridge_pieces(img[box], out[box], comp, n, affine @ shift, sp, levels, labels.names,
                                  max_mm=BRIDGE_SAME_MM)
    comp, n = ndi.label(out > 0, structure=np.ones((3, 3, 3)))
    bridged += _bridge_pieces(img, out, comp, n, affine, sp, levels, labels.names)
    return out, {"vessels": report, "bridged": bridged}


def _bridge_pieces(img, out, comp, n, affine, sp, levels, names, max_mm: float = BRIDGE_MM) -> list:
    """Fill a missed stretch of vessel where contrast runs from a cut-off piece to the largest one.

    ``comp`` numbers the pieces (1..n) that belong together; ``out`` and ``comp`` are changed in
    place. A piece is reconnected only along unlabelled voxels as bright as the lumen (and not
    brighter than a lumen can be, which keeps dense bone out), over a path of at most
    ``max_mm``. Only voxels lying between the two ends are filled.
    """
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    from skimage.graph import MCP_Geometric

    if n < 2:
        return []
    vox = float(np.prod(sp))
    sizes = np.bincount(comp.ravel())
    sizes[0] = 0
    main = int(np.argmax(sizes))
    boxes = ndi.find_objects(comp)

    def shell(i):
        box = boxes[i - 1]
        m = comp[box] == i
        return np.argwhere(m & ~ndi.binary_erosion(m)) + [s.start for s in box]

    pending = [i for i in range(1, n + 1) if i != main and sizes[i] * vox >= BRIDGE_MIN_MM3]
    if not pending:
        return []
    main_pts = shell(main)
    bridged = []
    while pending:
        tree = cKDTree(main_pts * sp)
        near = []
        for i in pending:
            pts = shell(i)
            d, j = tree.query(pts * sp)
            k = int(np.argmin(d))
            near.append((float(d[k]), i, pts[k], main_pts[int(j[k])], pts))
        gap, i, a, b, pts = min(near, key=lambda r: r[0])
        if gap > max_mm:
            break
        pending.remove(i)
        la = int(out[tuple(a)])
        if la not in levels or int(out[tuple(b)]) not in levels:
            continue
        level = 0.5 * (levels[la][0] + levels[int(out[tuple(b)])][0])
        cap = max(levels[la][1], levels[int(out[tuple(b)])][1])
        pad = np.ceil((max_mm + 2.0) / sp).astype(int)
        lo = np.maximum(np.minimum(a, b) - pad, 0)
        hi = np.minimum(np.maximum(a, b) + pad + 1, out.shape)
        box = tuple(slice(x, y) for x, y in zip(lo, hi))
        im, cb = img[box], comp[box]
        mine, theirs = cb == i, cb == main
        free = (im >= level) & (im <= cap) & (out[box] == 0)
        if not mine.any() or not theirs.any():
            continue
        seed = tuple(int(v) for v in a - lo)
        cost = np.full(im.shape, 1e6)
        cost[free | theirs] = 1.0
        cost[mine] = 1e-4                                   # distance is measured from the piece
        there, _ = MCP_Geometric(cost, sampling=tuple(float(s) for s in sp)).find_costs([seed])
        reach = np.where(theirs, there, np.inf)
        end = np.unravel_index(int(np.argmin(reach)), reach.shape)
        length = float(reach[end])
        if not np.isfinite(length) or length > max_mm:
            continue
        cost[mine], cost[theirs] = 1.0, 1e-4                # and back from the main tree
        back, _ = MCP_Geometric(cost, sampling=tuple(float(s) for s in sp)).find_costs([end])
        # every bright voxel lying between the two ends: the whole cross-section of the missed
        # stretch, but not a bright neighbour that only touches it
        fill = free & (there + back <= length + 1.0 + 0.1 * length)
        if not fill.any():
            continue
        out[box][fill] = la
        new = np.argwhere(fill) + lo
        cb[fill] = main
        comp[boxes[i - 1]][comp[boxes[i - 1]] == i] = main
        main_pts = np.vstack([main_pts, pts, new])
        mid = (affine @ np.r_[new.mean(0), 1.0])[:3]
        bridged.append({"vessel": names.get(la, str(la)), "length_mm": round(length, 1),
                        "volume_mm3": round(float(fill.sum()) * vox, 1),
                        "at_mm": [round(float(v), 1) for v in mid]})
    return bridged
