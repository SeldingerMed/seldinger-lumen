"""Scan reading, region detection, ask-before-install and naming, on generated data only."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")
pydicom = pytest.importorskip("pydicom")
nib = pytest.importorskip("nibabel")

from lumen.assets.scan_import import LabelVolume  # noqa: E402
from lumen.scan import pipeline, refine, region, segmenters  # noqa: E402
from lumen.scan.io import ScanError, ScanVolume, list_dicom_series, read_scan  # noqa: E402


# ---------------------------------------------------------------------------------------------
# DICOM written slice by slice with a known geometry
# ---------------------------------------------------------------------------------------------
def _write_series(folder, vol_ijk, origin, row, col, ps, step, *, modality="CT", uid="1.2.3",
                  shuffle=True, body_part="", skip=()):
    """vol_ijk[i, j, k]: i along the row direction (columns), j along the column direction."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, MRImageStorage

    folder.mkdir(parents=True, exist_ok=True)
    row, col = np.asarray(row, float), np.asarray(col, float)
    normal = np.cross(row, col)
    order = list(range(vol_ijk.shape[2]))
    if shuffle:
        np.random.default_rng(0).shuffle(order)
    for n, k in enumerate(order):
        if k in skip:
            continue
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage if modality == "CT" else MRImageStorage
        meta.MediaStorageSOPInstanceUID = f"{uid}.{k + 1}"
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"s{n:04d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.Modality, ds.SeriesInstanceUID, ds.StudyInstanceUID = modality, uid, "1.2"
        ds.SeriesDescription, ds.BodyPartExamined = "generated", body_part
        ds.PatientName, ds.PatientID = "Should^NotBeKept", "SECRET-ID"
        ds.Rows, ds.Columns = vol_ijk.shape[1], vol_ijk.shape[0]
        ds.PixelSpacing = [float(ps[0]), float(ps[1])]                # [between rows, between cols]
        ds.ImageOrientationPatient = [*map(float, row), *map(float, col)]
        ds.ImagePositionPatient = [float(v) for v in np.asarray(origin, float) + k * step * normal]
        ds.InstanceNumber = n + 1
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.PixelData = (vol_ijk[:, :, k].T + 1024).astype("<i2").tobytes()   # stored [row, col]
        ds.save_as(str(folder / f"s{n:04d}.dcm"), enforce_file_format=True)


def _ramp(shape):
    i, j, k = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return (3 * i + 7 * j + 11 * k - 500).astype(np.int16)


def test_dicom_series_geometry_values_and_no_identifiers(tmp_path):
    vol = _ramp((12, 9, 7))
    origin, ps, step = np.array([-30.0, 40.0, -200.0]), (0.8, 0.6), 1.5     # rows 0.8, cols 0.6
    _write_series(tmp_path / "ct", vol, origin, (1, 0, 0), (0, 1, 0), ps, step, body_part="CHEST")
    scan = read_scan(tmp_path / "ct")
    assert scan.modality == "CT" and scan.source == "dicom"
    assert scan.data.shape == vol.shape
    assert np.array_equal(scan.data, vol.astype(np.float32))                 # slope/intercept applied
    assert np.allclose(scan.spacing_mm, [0.6, 0.8, 1.5])
    # a voxel index maps to the millimetre position DICOM defines for it
    for ijk in ([0, 0, 0], [11, 0, 0], [0, 8, 0], [0, 0, 6], [5, 4, 3]):
        want = origin + np.array([ijk[0] * 0.6, ijk[1] * 0.8, ijk[2] * 1.5])
        assert np.allclose((scan.affine @ np.r_[ijk, 1.0])[:3], want, atol=1e-5)
    assert scan.hints["BodyPartExamined"] == "CHEST"
    assert "SECRET-ID" not in json.dumps(scan.hints) and "Should" not in json.dumps(scan.hints)


def test_dicom_oblique_orientation_maps_to_patient_mm(tmp_path):
    """A coronal-like series: voxel positions must follow the stated direction cosines."""
    vol = _ramp((10, 8, 6))
    row, col = np.array([1.0, 0, 0]), np.array([0, 0, -1.0])     # columns run left, rows run down
    origin = np.array([5.0, 6.0, 70.0])
    _write_series(tmp_path / "cor", vol, origin, row, col, (1.0, 1.0), 2.0)
    scan = read_scan(tmp_path / "cor")
    normal = np.cross(row, col)
    for ijk in ([0, 0, 0], [9, 7, 5], [3, 2, 4]):
        want = origin + ijk[0] * row + ijk[1] * col + ijk[2] * 2.0 * normal
        assert np.allclose((scan.affine @ np.r_[ijk, 1.0])[:3], want, atol=1e-5)
        assert scan.data[tuple(ijk)] == vol[tuple(ijk)]


def test_dicom_picks_largest_series_and_lists_them(tmp_path):
    _write_series(tmp_path / "a", _ramp((6, 6, 4)), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1), 1.0, uid="1.1")
    _write_series(tmp_path / "b", _ramp((6, 6, 9)), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1), 1.0, uid="1.9")
    rows = list_dicom_series(tmp_path)
    assert [r["slices"] for r in rows] == [9, 4]
    assert read_scan(tmp_path).data.shape[2] == 9
    assert read_scan(tmp_path, series="2").data.shape[2] == 4
    assert read_scan(tmp_path, series="1.1").data.shape[2] == 4


def test_dicom_missing_slice_and_tilt_are_refused(tmp_path):
    _write_series(tmp_path / "gap", _ramp((6, 6, 9)), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1), 1.0, skip=(4,))
    with pytest.raises(ScanError, match="not evenly spaced"):
        read_scan(tmp_path / "gap")
    with pytest.raises(ScanError, match="no CT or MR"):
        (tmp_path / "junk").mkdir()
        (tmp_path / "junk" / "notes.txt").write_text("hello")
        read_scan(tmp_path / "junk")


def test_nifti_and_nrrd_agree_on_patient_mm(tmp_path):
    nrrd = pytest.importorskip("nrrd")
    vol = _ramp((7, 6, 5)).astype(np.float32)
    lps = np.diag([0.5, 0.7, 2.0, 1.0])
    lps[:3, 3] = [-10.0, 20.0, 30.0]
    nib.save(nib.Nifti1Image(vol, np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps), str(tmp_path / "v.nii.gz"))
    nrrd.write(str(tmp_path / "v.nrrd"), vol, {"space": "left-posterior-superior",
               "space directions": lps[:3, :3].T, "space origin": lps[:3, 3]}, index_order="F")
    a, b = read_scan(tmp_path / "v.nii.gz"), read_scan(tmp_path / "v.nrrd")
    assert np.allclose(a.affine, lps, atol=1e-5) and np.allclose(b.affine, lps, atol=1e-5)
    assert np.array_equal(a.data, vol) and np.array_equal(b.data, vol)


# ---------------------------------------------------------------------------------------------
# region from the picture
# ---------------------------------------------------------------------------------------------
def _phantom(parts, spacing=(2.0, 2.0, 2.5), flip_z=False, modality="CT"):
    """parts: list of (kind, height_mm) from feet to head; kind in head / neck / trunk."""
    nx = ny = 220
    x = (np.arange(nx) - nx / 2) * spacing[0]
    X, Y = np.meshgrid(x, x, indexing="ij")
    slabs = []
    for kind, height in parts:
        s = np.full((nx, ny), -1000.0, np.float32)
        if kind == "head":
            r = np.hypot(X / 75.0, Y / 92.0)
            s[r <= 1.0] = 40.0
            s[(r > 0.9) & (r <= 1.0)] = 900.0                 # skull
        elif kind == "neck":
            s[np.hypot(X / 55.0, Y / 60.0) <= 1.0] = 40.0
            s[np.hypot(X, Y - 25.0) <= 14.0] = 600.0           # one vertebra
        else:
            s[np.hypot(X / 170.0, Y / 115.0) <= 1.0] = 40.0
            s[np.hypot(X, Y - 60.0) <= 20.0] = 600.0
        slabs += [s] * int(round(height / spacing[2]))
    data = np.stack(slabs, axis=2)
    A = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    if flip_z:
        data = data[:, :, ::-1]
        A[2, 2], A[2, 3] = -spacing[2], spacing[2] * (data.shape[2] - 1)
    return ScanVolume(np.ascontiguousarray(data), A, modality=modality)


@pytest.mark.parametrize("flip", [False, True])
def test_region_head_only(flip):
    r = region.detect_region(_phantom([("head", 170)], flip_z=flip))
    assert r.regions == ("head",) and r.basis == "picture" and r.confident


def test_region_trunk_only_even_when_the_header_says_head():
    scan = _phantom([("trunk", 400)])
    scan.hints = {"BodyPartExamined": "HEAD"}                   # a wrong tag must not win
    r = region.detect_region(scan)
    assert r.regions == ("body",) and r.basis == "picture"


@pytest.mark.parametrize("flip", [False, True])
def test_region_head_and_neck_finds_both_and_where_the_head_starts(flip):
    scan = _phantom([("trunk", 150), ("neck", 120), ("head", 180)], flip_z=flip)
    r = region.detect_region(scan)
    assert r.regions == ("head", "body")
    # the head starts 270 mm above the lowest slice; the cut sits a little below it
    assert 270 - 20 - 10 <= r.detail["head_from_mm"] <= 270


def test_region_falls_back_to_header_then_to_both():
    air = ScanVolume(np.full((40, 40, 40), -1000.0, np.float32), np.eye(4), modality="CT")
    r = region.detect_region(air)
    assert r.regions == ("head", "body") and not r.confident
    air.hints = {"StudyDescription": "CTA Circle of Willis"}
    r = region.detect_region(air)
    assert r.regions == ("head",) and r.basis == "header"
    air.hints = {"StudyDescription": "CT angio chest abdomen pelvis"}
    assert region.detect_region(air).regions == ("body",)


def test_mr_uses_the_header_when_it_speaks():
    scan = _phantom([("head", 170)], modality="MR")
    scan.data = np.where(scan.data > -500, 300.0, 0.0).astype(np.float32)
    scan.hints = {"SeriesDescription": "TOF 3D brain"}
    assert region.detect_region(scan).regions == ("head",)


# ---------------------------------------------------------------------------------------------
# ask before installing
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def lumen_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LUMEN_WEIGHTS", raising=False)
    return tmp_path / "home"


def _fake_install(calls):
    def install(seg, say):
        calls.append(seg.key)
        d = segmenters._dir(seg)
        py = segmenters._python(seg)
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("")
        (d / "installed.json").write_text(json.dumps({"commit": seg.commit}))
    return install


def test_prompt_names_model_size_licence_and_says_no_upload(lumen_home):
    text = segmenters.prompt_text(segmenters.TOTALSEGMENTATOR, "This scan shows the neck or trunk.")
    for needle in ("TotalSegmentator", "Apache-2.0", "GB", "SeldingerMed/TotalSegmentator",
                   "not sent anywhere", str(lumen_home), "[y/N]"):
        assert needle in text


def test_no_means_nothing_is_installed(lumen_home):
    calls, said = [], []
    with pytest.raises(segmenters.Declined):
        segmenters.ensure(segmenters.TOTALSEGMENTATOR, why="w", ask=lambda _: "n",
                          say=said.append, install=_fake_install(calls))
    assert calls == [] and not lumen_home.exists()
    with pytest.raises(segmenters.Declined):                      # empty answer is also no
        segmenters.ensure(segmenters.TOTALSEGMENTATOR, why="w", ask=lambda _: "",
                          install=_fake_install(calls))
    assert calls == []


def test_yes_installs_once_and_is_remembered(lumen_home):
    calls, asked = [], []

    def ask(text):
        asked.append(text)
        return "y"

    seg = segmenters.RSNA_INTRACRANIAL
    segmenters.ensure(seg, why="This scan shows the head.", ask=ask, say=lambda *_: None,
                      install=_fake_install(calls))
    assert calls == [seg.key] and len(asked) == 1 and segmenters.installed(seg)
    segmenters.ensure(seg, why="w", ask=ask, say=lambda *_: None, install=_fake_install(calls))
    assert calls == [seg.key] and len(asked) == 1                 # no second prompt, no reinstall
    assert not segmenters.installed(segmenters.TOTALSEGMENTATOR)  # the other model was not touched


def test_cannot_ask_without_a_terminal_unless_yes(lumen_home, monkeypatch):
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    calls = []
    with pytest.raises(segmenters.SegmenterError, match="--yes"):
        segmenters.ensure(segmenters.TOTALSEGMENTATOR, why="w", install=_fake_install(calls))
    assert calls == []
    segmenters.ensure(segmenters.TOTALSEGMENTATOR, why="w", assume_yes=True, say=lambda *_: None,
                      install=_fake_install(calls))
    assert calls == ["totalsegmentator"]


def test_a_pinned_commit_change_forces_a_fresh_install(lumen_home):
    calls = []
    seg = segmenters.TOTALSEGMENTATOR
    segmenters.ensure(seg, why="w", assume_yes=True, say=lambda *_: None, install=_fake_install(calls))
    (segmenters._dir(seg) / "installed.json").write_text(json.dumps({"commit": "older"}))
    assert not segmenters.installed(seg)


def test_weight_download_rejects_wrong_bytes(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc" * 100)
    good = segmenters._sha256(src)
    dest = tmp_path / "w" / "file.bin"
    segmenters._fetch(src.as_uri(), dest, 300, good)
    assert dest.read_bytes() == src.read_bytes()
    bad = tmp_path / "w" / "bad.bin"
    with pytest.raises(segmenters.SegmenterError, match="not the one Lumen was tested with"):
        segmenters._fetch(src.as_uri(), bad, 300, "0" * 64)
    assert not bad.exists() and not bad.with_name("bad.bin.part").exists()
    with pytest.raises(segmenters.SegmenterError):
        segmenters._fetch(src.as_uri(), bad, 200, good)            # larger than expected
    assert not bad.exists()


def test_routing_and_names():
    assert [s.key for s in segmenters.for_regions(("head",))] == ["rsna-intracranial"]
    assert [s.key for s in segmenters.for_regions(("body",))] == ["totalsegmentator"]
    assert [s.key for s in segmenters.for_regions(("head", "body"))] == ["totalsegmentator", "rsna-intracranial"]
    assert segmenters.pretty("common_carotid_artery_left") == "left common carotid artery"
    assert segmenters.pretty("aorta") == "aorta"


def test_mps_memory_cap_gets_a_valid_low_watermark(monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    env = segmenters._torch_env({"X": "1"})
    assert env["X"] == "1" and float(env["PYTORCH_MPS_LOW_WATERMARK_RATIO"]) < 0.5
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    assert "PYTORCH_MPS_LOW_WATERMARK_RATIO" not in segmenters._torch_env()


# ---------------------------------------------------------------------------------------------
# sharpening, naming, and the whole import with a stand-in segmenter
# ---------------------------------------------------------------------------------------------
def _fork_scan():
    """A CT-like Y fork: lumen 350 HU in 40 HU tissue, with fat labels drawn 1.5 mm too wide."""
    from tests.test_scan_import import _affine, _dist_to_polyline, _world_grid

    shape, A = (80, 60, 96), _affine((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    W = _world_grid(shape, A)[..., :3]
    polys = {1: (np.array([[40.0, 30, 4], [40.0, 30, 50]]), 7.0),
             2: (np.array([[40.0, 30, 50], [20.0, 30, 90]]), 4.0),
             3: (np.array([[40.0, 30, 50], [60.0, 30, 90]]), 4.0)}
    img = np.full(shape, 40.0, np.float32)
    fat = np.zeros(shape, np.int16)
    true = np.zeros(shape, bool)
    for value, (poly, r) in polys.items():
        d = _dist_to_polyline(W, poly)
        true |= d <= r
        fat[(d <= r + 1.5) & (fat == 0)] = value
    img[true] = 350.0
    from scipy import ndimage as ndi
    img = ndi.gaussian_filter(img, 0.7)
    names = {1: "aorta", 2: "left iliac artery", 3: "right iliac artery"}
    return ScanVolume(img, A, modality="CT"), LabelVolume(fat, A, names), true


def test_sharpening_pulls_fat_labels_back_to_the_lumen():
    scan, labels, true = _fork_scan()
    out, rep = refine.refine_labels(scan, labels)
    assert all(v["refined"] for v in rep.values())
    fat_extra = ((labels.data > 0) & ~true).sum()
    new_extra = ((out > 0) & ~true).sum()
    assert new_extra < 0.25 * fat_extra                           # most of the fat rim is gone
    assert ((out > 0) & true).sum() > 0.97 * true.sum()           # and the lumen is kept
    assert set(np.unique(out)) == {0, 1, 2, 3}


def test_sharpening_leaves_a_scan_without_contrast_alone():
    scan, labels, _ = _fork_scan()
    scan.data = np.full(scan.data.shape, 40.0, np.float32)
    out, rep = refine.refine_labels(scan, labels)
    assert np.array_equal(out, labels.data)
    assert not any(v["refined"] for v in rep.values())


def test_import_asks_segments_names_and_targets_by_name(tmp_path, lumen_home, monkeypatch):
    scan, labels, _ = _fork_scan()
    nib.save(nib.Nifti1Image(scan.data, np.diag([-1.0, -1.0, 1.0, 1.0]) @ scan.affine),
             str(tmp_path / "scan.nii.gz"))
    calls, asked = [], []
    monkeypatch.setattr(segmenters, "_install", _fake_install(calls))
    monkeypatch.setattr(segmenters, "run_body", lambda s, work, say=print: labels)
    monkeypatch.setattr(region, "detect_region", lambda s: region.RegionReport(("body",), True, "picture"))

    def ask(text):
        asked.append(text)
        return "yes"

    rep = pipeline.import_scan(tmp_path / "scan.nii.gz", tmp_path / "out", ask=ask, say=lambda *_: None)
    assert calls == ["totalsegmentator"] and len(asked) == 1 and "neck or trunk" in asked[0]
    assert rep["region"]["regions"] == ["body"]
    assert rep["segmentation"]["used"][0]["segmenter"] == "TotalSegmentator"
    assert set(rep["vessels"]) == {"aorta", "left iliac artery", "right iliac artery"}
    for f in ("asset.json", "surface.stl", "names.json", "report.json"):
        assert (tmp_path / "out" / f).exists()
    assert not (tmp_path / "out" / "work").exists()               # the scan copy is cleaned up
    asset, names = pipeline.load_imported(tmp_path / "out")
    node = pipeline.resolve_target(names, asset, "Left Iliac")
    assert node == names["vessels"]["left iliac artery"]["target_node"]
    tip = np.array(next(n.position_mm for n in asset.nodes if n.id == node))
    assert np.linalg.norm(tip - np.array([20.0, 30.0, 90.0])) < 8.0
    with pytest.raises(ValueError, match="several vessels"):
        pipeline.resolve_target(names, asset, "iliac")
    with pytest.raises(ValueError, match="no vessel or node"):
        pipeline.resolve_target(names, asset, "basilar")
    # sharpened walls: radii are the lumen's, not the fat labels'
    by = {e["name"]: e["median_radius_mm"] for e in rep["edges"]}
    assert abs(by["aorta"] - 7.0) < 0.6 and abs(by["left iliac artery"] - 4.0) < 0.6


def test_import_declined_writes_no_anatomy(tmp_path, lumen_home, monkeypatch):
    scan, labels, _ = _fork_scan()
    nib.save(nib.Nifti1Image(scan.data, np.diag([-1.0, -1.0, 1.0, 1.0]) @ scan.affine),
             str(tmp_path / "scan.nii.gz"))
    monkeypatch.setattr(region, "detect_region", lambda s: region.RegionReport(("body",), True, "picture"))
    with pytest.raises(segmenters.Declined, match="--mask"):
        pipeline.import_scan(tmp_path / "scan.nii.gz", tmp_path / "out", ask=lambda _: "n", say=lambda *_: None)
    assert not (tmp_path / "out" / "asset.json").exists()
