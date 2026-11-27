"""Work out which body region a scan covers, from the scan itself.

``lumen import`` uses this to choose a segmenter without asking the user: a scan that shows the
head goes to the intracranial model, one that shows the neck or trunk goes to the body model, and
one that shows both goes to both.

The decision comes from the picture first. The scan is walked slice by slice from the feet end to
the head end and each axial slice is measured (how big the body outline is, how wide it is, how
much bone it holds). A head is a run of slices about the size of a head with a skull's worth of
bone; anything else that is body counts as neck or trunk. Header text (body part, descriptions)
is used only when the picture is inconclusive, since such text is often missing or wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lumen.scan.io import is_ct

HEAD, BODY = "head", "body"

# axial slice measures, in cm2 / mm; chosen from open head, neck and aortic CTAs (see the tests)
_HEAD_AREA = (130.0, 300.0)       # outline area of a head slice
_HEAD_WIDTH = (135.0, 215.0)      # widest extent of a head slice
_HEAD_BONE = 20.0                 # a skull ring holds at least this much bone (CT)
_HEAD_RUN_MM = 50.0               # a head is at least this tall
_FACE_MM = 80.0                   # face and jaw below the skull run still belong to the head
_BODY_RUN_MM = 60.0               # neck or trunk worth segmenting is at least this tall
_SAMPLE_MM = 5.0

_HEAD_WORDS = ("HEAD", "BRAIN", "SKULL", "CEREBR", "INTRACRAN", "WILLIS", "COW", "NEURO", "CRANI")
_BODY_WORDS = ("NECK", "CAROTID", "CHEST", "THORA", "AORT", "ABD", "PELVI", "ILIAC", "RUNOFF",
               "CARDIAC", "HEART", "CORONAR", "LUNG", "RENAL", "BODY", "TRUNK")


@dataclass
class RegionReport:
    regions: tuple                    # subset of ("head", "body"), in that order
    confident: bool                   # False: nothing conclusive was found; try every segmenter
    basis: str                        # "picture", "header", "header and picture" or "none"
    head_mm: float = 0.0              # height of the head run found in the picture
    body_mm: float = 0.0              # height of neck/trunk found in the picture
    detail: dict = field(default_factory=dict)

    def __contains__(self, region: str) -> bool:
        return region in self.regions


def _axial_view(volume):
    """Return (array with axis 2 running feet -> head, in-plane spacing, slice spacing) or None
    when no array axis is close to the patient's long axis."""
    A = np.asarray(volume.affine, float)
    sp = np.linalg.norm(A[:3, :3], axis=0)
    Q = A[:3, :3] / sp
    ax = int(np.argmax(np.abs(Q[2])))
    if abs(Q[2, ax]) < 0.85:          # more than about 30 degrees off axial
        return None
    d = np.moveaxis(np.asarray(volume.data), ax, 2)
    if Q[2, ax] < 0:
        d = d[:, :, ::-1]
    return d, np.delete(sp, ax), float(sp[ax])


def _slice_measures(volume) -> dict | None:
    from scipy import ndimage as ndi

    view = _axial_view(volume)
    if view is None:
        return None
    d, inpl, dz = view
    ct = is_ct(volume)
    if ct:
        level = -300.0
    else:                              # MR: no absolute scale; body is the bright part
        level = 0.1 * float(np.percentile(d[::4, ::4, ::4], 99.5))
    step = max(1, int(round(_SAMPLE_MM / dz)))
    px = float(np.prod(inpl)) / 100.0  # cm2 per pixel
    small_fov = not ct and bool(np.all(np.asarray(d.shape[:2]) * inpl <= _HEAD_WIDTH[1]))
    area, bone, width, cut = [], [], [], []
    for k in range(0, d.shape[2], step):
        s = d[:, :, k]
        lab, n = ndi.label(s > level)
        if n == 0:
            area.append(0.0), bone.append(0.0), width.append(0.0), cut.append(False)
            continue
        sizes = np.bincount(lab.ravel())[1:]
        body = ndi.binary_fill_holes(lab == int(np.argmax(sizes)) + 1)
        ii, jj = np.nonzero(body)
        area.append(float(body.sum()) * px)
        bone.append(float(((s > 300.0) & body).sum()) * px if ct else 0.0)
        width.append(float(max((np.ptp(ii) + 1) * inpl[0], (np.ptp(jj) + 1) * inpl[1])))
        # the body runs off the picture on two or more sides: wider than the field of view. A
        # picture no wider than a head (an intracranial MRA slab, say) says nothing by this, so
        # there the outline's own size has to decide
        sides = [body[0].any(), body[-1].any(), body[:, 0].any(), body[:, -1].any()]
        cut.append(sum(sides) >= 2 and not small_fov)
    # patient z of the first (most caudal) slice of the feet-to-head view
    A = np.asarray(volume.affine, float)
    ax = int(np.argmax(np.abs(A[2, :3] / np.linalg.norm(A[:3, :3], axis=0))))
    z_ends = (A[2, 3], A[2, 3] + A[2, ax] * (np.asarray(volume.data).shape[ax] - 1))
    return {"area_cm2": np.asarray(area), "bone_cm2": np.asarray(bone), "width_mm": np.asarray(width),
            "cut_off": np.asarray(cut), "step_mm": step * dz, "ct": ct, "z0_mm": float(min(z_ends))}


def _longest_run(flags: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest run of True, allowing single-slice gaps."""
    f = np.asarray(flags, bool).copy()
    gap = ~f[1:-1] & f[:-2] & f[2:]
    f[1:-1] |= gap
    best, start, cur = (0, 0), 0, 0
    for i, v in enumerate(f):
        if v:
            if cur == 0:
                start = i
            cur += 1
            if cur > best[1]:
                best = (start, cur)
        else:
            cur = 0
    return best


def _from_header(hints: dict) -> tuple:
    text = " ".join(str(v) for v in (hints or {}).values()).upper()
    words = set(text.replace("_", " ").replace("-", " ").replace("/", " ").split())
    head = any(w in text for w in _HEAD_WORDS if w != "COW") or "COW" in words
    body = any(w in text for w in _BODY_WORDS)
    return tuple(r for r, hit in ((HEAD, head), (BODY, body)) if hit)


def detect_region(volume) -> RegionReport:
    """Decide which regions (``"head"``, ``"body"``) the scan covers."""
    m = _slice_measures(volume)
    header = _from_header(volume.hints)
    if m is None or not np.any(m["area_cm2"] > 0):
        if header:
            return RegionReport(header, True, "header", detail={"header": list(header)})
        return RegionReport((HEAD, BODY), False, "none")
    a, b, w, cut, step = m["area_cm2"], m["bone_cm2"], m["width_mm"], m["cut_off"], m["step_mm"]
    headlike = (a >= _HEAD_AREA[0]) & (a <= _HEAD_AREA[1]) & (w >= _HEAD_WIDTH[0]) \
        & (w <= _HEAD_WIDTH[1]) & ~cut
    if m["ct"]:
        headlike &= b >= _HEAD_BONE
    start, length = _longest_run(headlike)
    head_mm = length * step
    is_head = np.zeros(len(a), bool)
    head_from = None
    if head_mm >= _HEAD_RUN_MM:
        # the head is everything from the start of the skull run to the top of the scan, plus the
        # face and skull base just below it (head sized, but with less bone per slice)
        lo, floor = start, start - int(round(_FACE_MM / step))
        while lo > max(floor, 0) and a[lo - 1] >= _HEAD_AREA[0] \
                and _HEAD_WIDTH[0] <= w[lo - 1] <= _HEAD_WIDTH[1] and not cut[lo - 1]:
            lo -= 1
        is_head[lo:] = True
        head_from = lo
    body_mm = float(((a > 20.0) & ~is_head).sum() * step)
    has_head = bool(is_head.any())
    has_body = body_mm >= _BODY_RUN_MM
    detail = {"step_mm": step, "ct": bool(m["ct"]), "header": list(header)}
    if head_from is not None:
        # patient z (mm) of the lowest head slice, less a margin so the skull base is kept
        detail["head_from_mm"] = float(m["z0_mm"] + head_from * step - 20.0)
    if not m["ct"] and header:
        # MR has no bone signal to confirm a skull, so the header text counts as well as the
        # outline. Either one is enough to try a model: text such as "Carotid" on a scan that
        # also shows the head must not lose the head, and a model that finds nothing is harmless
        seen = {r for r, hit in ((HEAD, has_head), (BODY, has_body)) if hit} | set(header)
        regions = tuple(r for r in (HEAD, BODY) if r in seen)
        return RegionReport(regions, True, "header and picture" if len(seen) > len(header) else "header",
                            head_mm, body_mm, detail)
    if has_head or has_body:
        regions = tuple(r for r, hit in ((HEAD, has_head), (BODY, has_body)) if hit)
        return RegionReport(regions, True, "picture", head_mm, body_mm, detail)
    if header:
        return RegionReport(header, True, "header", head_mm, body_mm, detail)
    return RegionReport((HEAD, BODY), False, "none", head_mm, body_mm, detail)
