"""Read a cross-sectional scan (CTA or MRA) into one array with a patient-millimetre affine.

Accepted input: a folder of DICOM files (one or more series; classic single-frame CT/MR or
enhanced multi-frame), a single DICOM file, NIfTI (``.nii``, ``.nii.gz``) or NRRD (``.nrrd``).

Coordinates are the DICOM patient frame, LPS, in millimetres. Only the fields needed to pick a
segmenter are kept from the DICOM header (modality, body part, descriptions). No patient
identifiers are read into the result.

Needs the ``scan`` extra (pydicom / nibabel / pynrrd, only for the format actually opened).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
# header text used only to recognise the body region; never patient identifiers
_HINT_TAGS = ("BodyPartExamined", "StudyDescription", "SeriesDescription", "ProtocolName")
_VOLUME_MODALITIES = ("CT", "MR")


class ScanError(ValueError):
    """The input cannot be read as one cross-sectional volume."""


@dataclass
class ScanVolume:
    data: np.ndarray                 # (nx, ny, nz) float32; CT in Hounsfield units
    affine: np.ndarray               # 4x4: voxel index (i, j, k, 1) -> patient LPS mm
    modality: str = ""               # "CT", "MR" or "" when the file does not say
    hints: dict = field(default_factory=dict)     # region-hint header text (no identifiers)
    source: str = ""                 # "dicom", "nifti" or "nrrd"

    @property
    def spacing_mm(self) -> np.ndarray:
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def extent_mm(self) -> np.ndarray:
        return self.spacing_mm * np.asarray(self.data.shape)


def _header_modality(header, where) -> str:
    """'CT' or 'MR' when the file says so, else ''.

    NIfTI stores it in ``descrip`` (3D Slicer writes "Modality=CT"), NRRD in ``modality``.
    """
    if isinstance(header, dict):
        text = str(header.get("modality", ""))
    else:
        raw = header.get("descrip")
        text = (raw.tobytes() if hasattr(raw, "tobytes") else bytes(raw)).decode("latin1", "ignore")
    text = text.upper().replace("\x00", " ")
    if "MODALITY=CT" in text or text.strip() == "CT":
        return "CT"
    if "MODALITY=MR" in text or text.strip() == "MR":
        return "MR"
    return ""


def is_ct(scan: ScanVolume) -> bool:
    """CT by its header, or for a file with no modality (NIfTI, NRRD) by its values: air in
    Hounsfield units sits near -1000, and an MR image has no values that low."""
    if scan.modality:
        return scan.modality == "CT"
    return float(np.nanmin(scan.data)) <= -500.0


def read_scan(path: str | Path, series: str | None = None) -> ScanVolume:
    """Read a scan from a DICOM folder or file, a NIfTI file or an NRRD file.

    ``series`` selects one DICOM series by SeriesInstanceUID or by its 1-based position in
    ``list_dicom_series``; by default the series with the most slices is used.
    """
    p = Path(path)
    if not p.exists():
        raise ScanError(f"no such file or folder: {p}")
    low = p.name.lower()
    if p.is_dir():
        return _read_dicom(p, series)
    if low.endswith(".nii") or low.endswith(".nii.gz"):
        return _read_nifti(p)
    if low.endswith(".nrrd"):
        return _read_nrrd(p)
    return _read_dicom(p, series)


def _check_affine(aff: np.ndarray, where) -> None:
    sp = np.linalg.norm(aff[:3, :3], axis=0)
    if not np.all(np.isfinite(aff)) or not np.all(sp > 1e-6):
        raise ScanError(f"the scan geometry is not usable: {where}")


def _read_nifti(p: Path) -> ScanVolume:
    import nibabel as nib

    img = nib.load(str(p))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ScanError(f"expected one 3-D volume, got shape {data.shape}: {p}")
    aff = _RAS_TO_LPS @ np.asarray(img.affine, float)
    _check_affine(aff, p)
    return ScanVolume(data, aff, modality=_header_modality(img.header, p), source="nifti")


def _read_nrrd(p: Path) -> ScanVolume:
    import nrrd

    data, hdr = nrrd.read(str(p), index_order="F")
    if data.ndim != 3:
        raise ScanError(f"expected one 3-D volume, got shape {data.shape}: {p}")
    if "space directions" not in hdr or "space origin" not in hdr:
        raise ScanError(f"the NRRD file carries no patient geometry: {p}")
    aff = np.eye(4)
    aff[:3, :3] = np.asarray(hdr["space directions"], float).T
    aff[:3, 3] = np.asarray(hdr["space origin"], float)
    space = str(hdr.get("space", "left-posterior-superior")).lower()
    if space in ("right-anterior-superior", "ras"):
        aff = _RAS_TO_LPS @ aff
    elif space not in ("left-posterior-superior", "lps"):
        raise ScanError(f"unsupported NRRD space {space!r}: {p}")
    _check_affine(aff, p)
    return ScanVolume(np.asarray(data, dtype=np.float32), aff,
                      modality=_header_modality(hdr, p), source="nrrd")


# ---------------------------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------------------------
def _dicom_files(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    files = [f for f in sorted(p.rglob("*")) if f.is_file() and not f.name.startswith(".")
             and f.name.upper() != "DICOMDIR"]
    if not files:
        raise ScanError(f"the folder is empty: {p}")
    return files


def _headers(p: Path) -> dict:
    """Group readable volumetric slices by series: {uid: [(path, header), ...]}."""
    import pydicom
    from pydicom.errors import InvalidDicomError

    out: dict = {}
    for f in _dicom_files(p):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except (InvalidDicomError, OSError, ValueError, AttributeError):
            continue
        if str(getattr(ds, "Modality", "")) not in _VOLUME_MODALITIES:
            continue
        if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
            continue
        out.setdefault(str(getattr(ds, "SeriesInstanceUID", "")), []).append((f, ds))
    if not out:
        raise ScanError(f"no CT or MR DICOM slices found in {p} (angiography and other "
                        "projection images are not cross-sectional scans)")
    return out


def _n_slices(items) -> int:
    return sum(int(getattr(ds, "NumberOfFrames", 1) or 1) for _, ds in items)


def list_dicom_series(path: str | Path) -> list[dict]:
    """One row per CT/MR series in a folder, largest first."""
    rows = []
    for uid, items in _headers(Path(path)).items():
        ds = items[0][1]
        rows.append({"series": uid, "modality": str(ds.Modality), "slices": _n_slices(items),
                     "description": str(getattr(ds, "SeriesDescription", ""))})
    rows.sort(key=lambda r: (-r["slices"], r["series"]))
    return rows


def _read_dicom(p: Path, series: str | None) -> ScanVolume:
    groups = _headers(p)
    order = sorted(groups, key=lambda u: (-_n_slices(groups[u]), u))
    if series is None:
        uid = order[0]
    elif str(series) in groups:                   # a SeriesInstanceUID wins over a list number
        uid = str(series)
    elif str(series).isdigit() and 1 <= int(series) <= len(order):
        uid = order[int(series) - 1]
    else:
        raise ScanError(f"no series {series!r} in {p}; found {len(order)} series (give a "
                        f"SeriesInstanceUID or a number from --list-series)")
    items = groups[uid]
    first = items[0][1]
    hints = {k: str(getattr(first, k, "") or "") for k in _HINT_TAGS}
    modality = str(first.Modality)
    multi = [int(getattr(h, "NumberOfFrames", 1) or 1) > 1 for _, h in items]
    if any(multi):
        if not all(multi):
            raise ScanError(f"one series mixes multi-frame and single-frame files: {p}")
        # an enhanced series may be split over several multi-frame files: gather every frame
        positions, frames, ref = [], [], None
        for f, _ in items:
            pos_f, frames_f, ref_f = _enhanced_frames(f)
            if ref is not None and (np.abs(np.r_[ref_f[0], ref_f[1]] - np.r_[ref[0], ref[1]]).max() > 1e-3
                                    or np.abs(np.asarray(ref_f[3]) - ref[3]).max() > 1e-4
                                    or frames_f[0].shape != frames[0].shape):
                raise ScanError(f"files of one series differ in orientation, spacing or size: {p}")
            ref = ref or ref_f
            positions += pos_f
            frames += frames_f
        data, aff = _stack(positions, ref[0], ref[1], ref[2], ref[3], frames, p)
    else:
        data, aff = _classic_volume(items)
    _check_affine(aff, p)
    return ScanVolume(data, aff, modality=modality, hints=hints, source="dicom")


def _pixels(path: Path):
    import pydicom

    ds = pydicom.dcmread(str(path))
    try:
        return ds, ds.pixel_array
    except Exception as exc:            # a compressed transfer syntax with no decoder installed
        raise ScanError(f"cannot decode the pixel data of {path.name} ({exc}); for compressed "
                        "DICOM install a decoder such as pylibjpeg or gdcm") from exc


def _stack(positions, row, col, normal, ps, frames, where) -> tuple[np.ndarray, np.ndarray]:
    """Order frames along the slice normal and build the volume and its affine."""
    d = np.asarray([float(np.dot(q, normal)) for q in positions])
    order = np.argsort(d)
    d = d[order]
    gaps = np.diff(d)
    if len(d) < 3:
        raise ScanError(f"a volume needs at least three slices, found {len(d)}: {where}")
    if np.any(gaps < 1e-4):
        raise ScanError(f"two slices share one position (several volumes in one series?): {where}")
    step = float(np.median(gaps))
    if np.abs(gaps - step).max() > max(0.02 * step, 0.01):
        raise ScanError(f"the slices are not evenly spaced (missing slices or gantry tilt): {where}")
    p0 = np.asarray(positions[order[0]], float)
    p_last = np.asarray(positions[order[-1]], float)
    k_dir = (p_last - p0) / (len(d) - 1)
    if np.linalg.norm(np.cross(k_dir, normal)) > 0.02 * step:
        raise ScanError(f"the slice stack is sheared (gantry tilt is not supported): {where}")
    aff = np.eye(4)
    # DICOM: pixel (row r, column c) sits at position + c * PixelSpacing[1] * row_cosine
    #        + r * PixelSpacing[0] * col_cosine; our array index is (i=c, j=r, k=slice)
    aff[:3, 0] = np.asarray(row, float) * ps[1]
    aff[:3, 1] = np.asarray(col, float) * ps[0]
    aff[:3, 2] = k_dir
    aff[:3, 3] = p0
    vol = np.stack([frames[i] for i in order], axis=-1)      # (rows, cols, k)
    return np.ascontiguousarray(vol.transpose(1, 0, 2)), aff


def _orientation(iop, where):
    v = np.asarray([float(x) for x in iop])
    if v.shape != (6,):
        raise ScanError(f"missing slice orientation: {where}")
    row, col = v[:3], v[3:]
    if abs(np.dot(row, col)) > 1e-3 or abs(np.linalg.norm(row) - 1) > 1e-3 \
            or abs(np.linalg.norm(col) - 1) > 1e-3:
        raise ScanError(f"invalid slice orientation: {where}")
    return row, col, np.cross(row, col)


def _classic_volume(items) -> tuple[np.ndarray, np.ndarray]:
    where = items[0][0].parent
    ref = items[0][1]
    for name in ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing"):
        if not hasattr(ref, name):
            raise ScanError(f"the slices carry no {name}; not a patient-space volume: {where}")
    row, col, normal = _orientation(ref.ImageOrientationPatient, where)
    ps = [float(x) for x in ref.PixelSpacing]
    positions, frames = [], []
    for f, hdr in items:
        for name in ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing", "Rows", "Columns"):
            if not hasattr(hdr, name):
                raise ScanError(f"a slice carries no {name}; the series cannot be stacked: {f}")
        if (int(hdr.Rows), int(hdr.Columns)) != (int(ref.Rows), int(ref.Columns)):
            raise ScanError(f"slices of different size in one series: {where}")
        iop = np.asarray([float(x) for x in hdr.ImageOrientationPatient])
        if np.abs(iop - np.r_[row, col]).max() > 1e-3:
            raise ScanError(f"slices of different orientation in one series: {where}")
        if np.abs(np.asarray([float(x) for x in hdr.PixelSpacing]) - ps).max() > 1e-4:
            raise ScanError(f"slices of different pixel spacing in one series: {where}")
        ds, px = _pixels(f)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
        frames.append(px.astype(np.float32) * slope + inter)
        positions.append([float(x) for x in hdr.ImagePositionPatient])
    return _stack(positions, row, col, normal, ps, frames, where)


def _enhanced_frames(path: Path):
    """Positions, frames and (row, col, normal, spacing) of one enhanced multi-frame file."""
    ds, px = _pixels(path)
    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    per = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if not per or len(per) != px.shape[0]:
        raise ScanError(f"the multi-frame file has no per-frame geometry: {path}")
    common = shared[0] if shared else None          # every group may also live per frame

    def group(frame, name):
        seq = getattr(frame, name, None) or (getattr(common, name, None) if common is not None else None)
        return seq[0] if seq else None

    positions, frames, ref = [], [], None
    for i, frame in enumerate(per):
        pos, ori, meas = (group(frame, n) for n in
                          ("PlanePositionSequence", "PlaneOrientationSequence", "PixelMeasuresSequence"))
        tags = ((pos, "ImagePositionPatient"), (ori, "ImageOrientationPatient"), (meas, "PixelSpacing"))
        for item, tag in tags:
            if item is None or not hasattr(item, tag):
                raise ScanError(f"frame {i + 1} of the multi-frame file carries no {tag}: {path}")
        row, col, normal = _orientation(ori.ImageOrientationPatient, path)
        ps = [float(x) for x in meas.PixelSpacing]
        if ref is None:
            ref = (row, col, normal, ps)
        elif np.abs(np.r_[row, col] - np.r_[ref[0], ref[1]]).max() > 1e-3 \
                or np.abs(np.asarray(ps) - ref[3]).max() > 1e-4:
            raise ScanError(f"frames of different orientation or spacing in one file: {path}")
        tr = group(frame, "PixelValueTransformationSequence")
        slope = float(getattr(tr, "RescaleSlope", 1) or 1) if tr is not None else 1.0
        inter = float(getattr(tr, "RescaleIntercept", 0) or 0) if tr is not None else 0.0
        frames.append(px[i].astype(np.float32) * slope + inter)
        positions.append([float(x) for x in pos.ImagePositionPatient])
    return positions, frames, ref
