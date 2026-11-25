"""Sharpen a segmenter's artery labels against the scan they came from.

Segmentation models work on a coarse grid (TotalSegmentator at 1.5 mm) and tend to draw a vessel
slightly fat: the wall and a rim of partial-volume voxels are included. For navigation the
surface that matters is the contrast-filled lumen. In a CTA or a bright-blood MRA that lumen is
visible in the picture itself, so each labelled vessel is redrawn at the half-way brightness
between its own lumen and the tissue around it, staying within a thin band of the model's
outline so that nearby bone or veins cannot leak in.

A scan with no bright lumen (no contrast) is left as the model drew it.
"""

from __future__ import annotations

import numpy as np

BAND_MM = 2.0            # how far the refined wall may move from the model's outline
MIN_CONTRAST = 60.0      # CT: lumen must be this many HU brighter than its surroundings


def _ball(radius_mm: float, spacing) -> np.ndarray:
    r = [max(1, int(np.ceil(radius_mm / s))) for s in spacing]
    g = np.meshgrid(*[np.arange(-n, n + 1) * s for n, s in zip(r, spacing)], indexing="ij")
    return (g[0] ** 2 + g[1] ** 2 + g[2] ** 2) <= radius_mm ** 2 + 1e-9


def refine_labels(scan, labels, band_mm: float = BAND_MM) -> tuple[np.ndarray, dict]:
    """Return ``(refined label array, report)``; the input label volume is not modified.

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
    ct = scan.modality == "CT" or (not scan.modality and float(img.min()) <= -500.0)
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
    return out, report
