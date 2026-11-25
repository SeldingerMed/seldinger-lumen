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


def test_mr_header_text_does_not_lose_what_the_picture_shows():
    # a head-and-neck MRA described only as "Carotid": the head in the picture still counts
    scan = _phantom([("trunk", 200), ("neck", 80), ("head", 170)], modality="MR")
    scan.data = np.where(scan.data > -500, 300.0, 0.0).astype(np.float32)
    scan.hints = {"SeriesDescription": "MRA Carotid"}
    r = region.detect_region(scan)
    assert r.regions == ("head", "body") and r.basis == "header and picture"
    # and with no header text at all the picture alone still decides
    scan.hints = {}
    assert region.detect_region(scan).regions == ("head", "body")


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
        segmenters._fetch(src.as_uri(), bad, 200, good)            # size mismatch is rejected
    assert not bad.exists() and not bad.with_name("bad.bin.part").exists()


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
    assert all(v["refined"] for v in rep["vessels"].values()) and rep["bridged"] == []
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
    assert not any(v["refined"] for v in rep["vessels"].values())


def test_a_stretch_the_model_missed_is_filled_where_the_scan_shows_contrast():
    scan, labels, true = _fork_scan()
    z = np.arange(labels.data.shape[2])[None, None, :]
    cut = (labels.data == 1) & (z >= 22) & (z <= 30)              # the model missed 9 mm of aorta
    holed = LabelVolume(np.where(cut, 0, labels.data), labels.affine, labels.names)
    out, rep = refine.refine_labels(scan, holed)
    assert len(rep["bridged"]) == 1
    b = rep["bridged"][0]
    # sharpening already recovers up to 2 mm at each cut end, so 5 to 9 mm is left to fill
    assert b["vessel"] == "aorta" and 4.5 <= b["length_mm"] <= 9.5 and 20.0 <= b["at_mm"][2] <= 32.0
    from scipy import ndimage as ndi
    assert ndi.label(out > 0, structure=np.ones((3, 3, 3)))[1] == 1  # one connected tree again
    assert ((out > 0) & ~true).sum() < 0.02 * true.sum()             # nothing filled outside the lumen
    assert (out[40, 30, 22:31] == 1).all()                           # the axis of the gap is aorta

    # a vessel missed where its far part is still attached by another route (a loop, as in the
    # circle of Willis) is mended too: a named vessel should be one piece
    from tests.test_scan_import import _dist_to_polyline, _world_grid

    Wd = _world_grid(labels.data.shape, labels.affine)[..., :3]
    bar = _dist_to_polyline(Wd, np.array([[20.0, 30, 86], [60.0, 30, 86]])) <= 3.0
    loop_scan = ScanVolume(np.where(bar, 350.0, scan.data).astype(np.float32), scan.affine, modality="CT")
    looped = np.where(bar & (labels.data == 0), 3, labels.data)
    side = (looped == 2) & (z >= 62) & (z <= 69)                  # 8 slices of the left iliac missed
    holed2 = LabelVolume(np.where(side, 0, looped), labels.affine, labels.names)
    assert ndi.label(holed2.data > 0, structure=np.ones((3, 3, 3)))[1] == 1   # still one tree
    assert ndi.label(holed2.data == 2, structure=np.ones((3, 3, 3)))[1] == 2  # but a broken vessel
    out3, rep3 = refine.refine_labels(loop_scan, holed2)
    assert [b["vessel"] for b in rep3["bridged"]] == ["left iliac artery"]
    assert ndi.label(out3 == 2, structure=np.ones((3, 3, 3)))[1] == 1
    assert ((out3 > 0) & ~(true | bar)).sum() < 0.02 * true.sum()

    # the same hole with no contrast across it stays a hole
    dark = ScanVolume(np.where(cut, 40.0, scan.data).astype(np.float32), scan.affine, modality="CT")
    out2, rep2 = refine.refine_labels(dark, holed)
    assert rep2["bridged"] == [] and ndi.label(out2 > 0, structure=np.ones((3, 3, 3)))[1] == 2


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


def test_head_and_neck_scan_asks_for_both_models_before_any_work(tmp_path, lumen_home, monkeypatch):
    scan, labels, _ = _fork_scan()
    nib.save(nib.Nifti1Image(scan.data, np.diag([-1.0, -1.0, 1.0, 1.0]) @ scan.affine),
             str(tmp_path / "scan.nii.gz"))
    events = []

    def install(seg, say):
        events.append(("install", seg.key))
        _fake_install([])(seg, say)

    def run_body(s, work, say=print):
        assert segmenters.installed(segmenters.TOTALSEGMENTATOR)
        events.append(("run", "body"))
        return labels

    def run_head(s, work, say=print, head_from_mm=None):
        assert segmenters.installed(segmenters.RSNA_INTRACRANIAL)
        events.append(("run", "head"))
        raise segmenters.SegmenterError("no intracranial arteries in this scan")

    def ask(text):
        events.append(("ask", "RSNA" if "RSNA" in text else "TotalSegmentator"))
        return "y"

    monkeypatch.setattr(segmenters, "_install", install)
    monkeypatch.setattr(segmenters, "run_body", run_body)
    monkeypatch.setattr(segmenters, "run_head", run_head)
    both = region.RegionReport(("head", "body"), True, "picture")
    monkeypatch.setattr(region, "detect_region", lambda s: both)
    said = []
    rep = pipeline.import_scan(tmp_path / "scan.nii.gz", tmp_path / "out", ask=ask, say=said.append)
    asks = [i for i, e in enumerate(events) if e[0] == "ask"]
    runs = [i for i, e in enumerate(events) if e[0] == "run"]
    assert [events[i][1] for i in asks] == ["TotalSegmentator", "RSNA"] and max(asks) < min(runs)
    assert sorted(e[1] for e in events if e[0] == "install") == ["rsna-intracranial", "totalsegmentator"]
    assert any("in the background" in s for s in said)
    assert sum("This scan shows" in s for s in said) == 0          # said once, in the question
    assert [u["segmenter"] for u in rep["segmentation"]["used"]] == ["TotalSegmentator"]
    assert "no intracranial arteries" in rep["segmentation"]["notes"][0]

    # second import: both are installed, so nothing is asked and nothing is installed again
    events.clear()
    pipeline.import_scan(tmp_path / "scan.nii.gz", tmp_path / "out2", ask=ask, say=said.append)
    assert [e[0] for e in events] == ["run", "run"]


def test_saying_no_to_one_of_two_models_still_imports_with_the_other(tmp_path, lumen_home, monkeypatch):
    scan, labels, _ = _fork_scan()
    nib.save(nib.Nifti1Image(scan.data, np.diag([-1.0, -1.0, 1.0, 1.0]) @ scan.affine),
             str(tmp_path / "scan.nii.gz"))
    calls = []
    monkeypatch.setattr(segmenters, "_install", _fake_install(calls))
    monkeypatch.setattr(segmenters, "run_body", lambda s, work, say=print: labels)
    both = region.RegionReport(("head", "body"), True, "picture")
    monkeypatch.setattr(region, "detect_region", lambda s: both)
    rep = pipeline.import_scan(tmp_path / "scan.nii.gz", tmp_path / "out", say=lambda *_: None,
                               ask=lambda text: "n" if "RSNA" in text else "y")
    assert calls == ["totalsegmentator"] and not segmenters.installed(segmenters.RSNA_INTRACRANIAL)
    assert "was not installed" in rep["segmentation"]["notes"][0]
    assert (tmp_path / "out" / "asset.json").exists()
    told = "\n".join(pipeline.summary_lines(rep, tmp_path / "out"))
    assert "note: the RSNA 2025 intracranial artery model was not installed" in told


def test_a_failed_background_install_is_reported_not_swallowed(tmp_path, lumen_home, monkeypatch):
    scan, labels, _ = _fork_scan()
    nib.save(nib.Nifti1Image(scan.data, np.diag([-1.0, -1.0, 1.0, 1.0]) @ scan.affine),
             str(tmp_path / "scan.nii.gz"))

    def broken(seg, say):
        raise OSError("disk full")

    monkeypatch.setattr(segmenters, "_install", broken)
    monkeypatch.setattr(region, "detect_region", lambda s: region.RegionReport(("body",), True, "picture"))
    with pytest.raises(segmenters.SegmenterError, match="disk full"):
        pipeline.import_scan(tmp_path / "scan.nii.gz", tmp_path / "out", assume_yes=True, say=lambda *_: None)
    assert not (tmp_path / "out" / "asset.json").exists()


def test_install_is_skipped_when_another_process_already_finished_it(lumen_home, monkeypatch):
    seg = segmenters.RSNA_INTRACRANIAL
    _fake_install([])(seg, None)                                   # as left by the other process
    monkeypatch.setattr(segmenters, "_install_locked",
                        lambda *a, **k: pytest.fail("installed twice"))
    segmenters._install(seg, lambda *_: None)
    (segmenters._dir(seg) / "installed.json").unlink()             # not installed: it does run
    ran = []
    monkeypatch.setattr(segmenters, "_install_locked", lambda s, say=print: ran.append(s.key))
    segmenters._install(seg, lambda *_: None)
    assert ran == [seg.key]


def test_a_hung_step_is_stopped_and_reported(tmp_path):
    import sys

    with pytest.raises(segmenters.SegmenterError, match="took longer than"):
        segmenters._run([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "log.txt",
                        what="running the model", timeout=0.5)
    with pytest.raises(segmenters.SegmenterError, match=r"failed \(exit 3\)"):
        segmenters._run([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"],
                        tmp_path / "log.txt", what="running the model")
    assert "boom" in (tmp_path / "log.txt").read_text()


def test_an_install_folder_that_cannot_hold_an_environment_is_refused_before_asking(
        lumen_home, monkeypatch):
    seg = segmenters.TOTALSEGMENTATOR
    monkeypatch.setattr(segmenters, "_filesystem", lambda path: "exfat")
    asked = []
    with pytest.raises(segmenters.SegmenterError, match="formatted as exfat") as err:
        segmenters.agree(seg, why="w", ask=lambda text: asked.append(text) or "y")
    assert asked == [] and "LUMEN_WEIGHTS" in str(err.value) and "Declined" not in type(err.value).__name__
    monkeypatch.setattr(segmenters, "_filesystem", lambda path: "apfs")
    segmenters.agree(seg, why="w", ask=lambda text: "y")
    monkeypatch.setattr(segmenters, "_filesystem", lambda path: "")     # cannot tell: carry on
    segmenters.agree(seg, why="w", ask=lambda text: "y")


def test_filesystem_probe_reads_a_folder_that_does_not_exist_yet(tmp_path):
    import sys

    fs = segmenters._filesystem(tmp_path / "not" / "made" / "yet")
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        assert fs.isalnum() and fs == segmenters._filesystem(tmp_path)   # a type name: apfs, ext4
    else:
        assert fs == ""


@pytest.mark.parametrize("damaged", ["[]", '"yes"', '{"totalsegmentator": true}', "{not json"])
def test_a_damaged_record_file_means_ask_again_not_a_crash(lumen_home, damaged):
    seg = segmenters.TOTALSEGMENTATOR
    segmenters._consent_file().parent.mkdir(parents=True, exist_ok=True)
    segmenters._consent_file().write_text(damaged)
    segmenters._dir(seg).mkdir(parents=True, exist_ok=True)
    (segmenters._dir(seg) / "installed.json").write_text(damaged)
    assert not segmenters.installed(seg)
    asked = []
    with pytest.raises(segmenters.Declined):
        segmenters.agree(seg, why="w", ask=lambda text: asked.append(text) or "n")
    assert len(asked) == 1                                         # true is not a recorded yes


def test_a_download_cut_short_leaves_no_partial_file(tmp_path, monkeypatch):
    import http.client
    import urllib.request

    class Resp:
        def __init__(self):
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, k):
            self.n += 1
            if self.n == 1:
                return b"x" * 100
            raise http.client.IncompleteRead(b"")                  # not an OSError

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: Resp())
    dest = tmp_path / "w" / "file.bin"
    with pytest.raises(segmenters.SegmenterError, match="could not download file.bin"):
        segmenters._fetch("https://example.invalid/file.bin", dest, 300, "0" * 64)
    assert list((tmp_path / "w").iterdir()) == []


def test_a_stuck_question_to_the_model_environment_is_stopped(lumen_home, monkeypatch):
    import sys

    seg = segmenters.TOTALSEGMENTATOR
    monkeypatch.setattr(segmenters, "_python", lambda s: sys.executable)
    monkeypatch.setattr(segmenters, "PROBE_TIMEOUT_S", 0.5)
    with pytest.raises(segmenters.SegmenterError, match="took longer than"):
        segmenters._ask_env(seg, "import time; time.sleep(30)", "checking the processor")
    assert segmenters._ask_env(seg, "print('ok')", "checking").stdout.strip() == "ok"


def test_a_scan_and_a_mask_together_are_refused(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        pipeline.import_scan(tmp_path / "scan.nii.gz", tmp_path / "out", mask=tmp_path / "m.nii.gz")
    with pytest.raises(ValueError, match="give a scan"):
        pipeline.import_scan(None, tmp_path / "out")
    assert not (tmp_path / "out").exists()                         # refused before writing anything


def test_the_head_model_environment_is_held_to_numpy_2(lumen_home, monkeypatch):
    asked = []
    monkeypatch.setattr(segmenters, "_make_env", lambda seg, log: None)
    monkeypatch.setattr(segmenters, "_pip", lambda seg, args, log: asked.extend(args))
    monkeypatch.setattr(segmenters, "_fetch", lambda *a, **k: None)
    segmenters._install(segmenters.RSNA_INTRACRANIAL, lambda *_: None)
    assert "numpy>=2.0" in asked and any(a.startswith("nnxnet @ ") for a in asked)


@pytest.mark.parametrize("argv, said", [
    (["--mask", "m.nii.gz", "--list-series"], "--list-series needs a scan"),
    (["scan-folder", "--mask", "m.nii.gz"], "not both"),
    ([], "give a scan or --mask"),
])
def test_the_command_refuses_arguments_that_do_not_go_together(argv, said, capsys):
    from lumen import cli

    with pytest.raises(SystemExit) as stop:
        cli.import_scan_main(argv, prog="lumen import")
    assert stop.value.code == 2 and said in capsys.readouterr().err


def test_without_uv_the_model_environment_still_gets_its_own_python(lumen_home, monkeypatch):
    import shutil
    import sys

    seg = segmenters.TOTALSEGMENTATOR
    ran = []
    monkeypatch.setattr(segmenters, "_run", lambda cmd, log, what="": ran.append(cmd))
    other = "3.11" if sys.version_info[:2] != (3, 11) else "3.10"
    odd = segmenters.Segmenter(**{**seg.__dict__, "python": other})
    # uv missing, the model's Python on the path: that one is used, not the one running Lumen
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "uv" else f"/somewhere/{name}")
    segmenters._make_env(odd, lumen_home / "log")
    assert ran[-1][0] == f"/somewhere/python{other}"
    # neither there: stop and say what to install, never build on an untried Python
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(segmenters.SegmenterError, match=f"needs Python {other}.*Install uv"):
        segmenters._make_env(odd, lumen_home / "log")
    # Lumen already on the model's Python: use it
    same = segmenters.Segmenter(**{**seg.__dict__, "python": "{}.{}".format(*sys.version_info[:2])})
    segmenters._make_env(same, lumen_home / "log")
    assert ran[-1][0] == sys.executable


def test_a_vessel_the_wire_cannot_reach_is_refused_not_swapped(tmp_path):
    names = {"vessels": {"left vertebral artery": {"edges": ["e9"], "target_node": None}}}

    class A:
        nodes = []

    with pytest.raises(ValueError, match="no node the wire can reach"):
        pipeline.resolve_target(names, A(), "vertebral")


def test_merge_lists_only_vessels_that_kept_voxels():
    from lumen.assets.scan_import import LabelVolume

    a = np.zeros((4, 4, 4), np.int16)
    a[1:3, 1:3, 1:3] = 1
    b = np.zeros_like(a)
    b[1:3, 1:3, 1:3] = 1                                       # entirely under the first model's label
    b[0, 0, 0] = 2
    m = pipeline._merge([LabelVolume(a, np.eye(4), {1: "aorta"}),
                         LabelVolume(b, np.eye(4), {1: "aorta copy", 2: "left iliac artery"})])
    assert sorted(m.names.values()) == ["aorta", "left iliac artery"]
    assert set(np.unique(m.data)) - {0} == set(m.names)


def test_a_slice_missing_its_geometry_is_a_clean_error(tmp_path):
    import pydicom

    _write_series(tmp_path / "ct", _ramp((6, 6, 5)), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1), 1.0, shuffle=False)
    f = sorted((tmp_path / "ct").iterdir())[3]
    ds = pydicom.dcmread(f)
    del ds.ImagePositionPatient
    ds.save_as(f)
    with pytest.raises(ScanError, match="ImagePositionPatient"):
        read_scan(tmp_path / "ct")


def _write_multiframe(f, vol, z0, *, uid="1.5", drop=None):
    """Enhanced CT, all geometry per frame and an empty shared group (as some scanners write)."""
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2.1"           # enhanced CT
    meta.MediaStorageSOPInstanceUID = f"{uid}.{z0}"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(f), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.SeriesInstanceUID, ds.StudyInstanceUID = "CT", uid, "1.2"
    ds.Rows, ds.Columns, ds.NumberOfFrames = vol.shape[1], vol.shape[0], vol.shape[2]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.SharedFunctionalGroupsSequence = Sequence([])
    frames = []
    for k in range(vol.shape[2]):
        fr = Dataset()
        pos, ori, meas, tr = Dataset(), Dataset(), Dataset(), Dataset()
        pos.ImagePositionPatient = [0.0, 0.0, 2.0 * (z0 + k)]
        ori.ImageOrientationPatient = [1.0, 0, 0, 0, 1.0, 0]
        meas.PixelSpacing = [1.0, 1.0]
        tr.RescaleSlope, tr.RescaleIntercept = 1, -1024
        if drop == k:
            del pos.ImagePositionPatient
        fr.PlanePositionSequence, fr.PlaneOrientationSequence = Sequence([pos]), Sequence([ori])
        fr.PixelMeasuresSequence, fr.PixelValueTransformationSequence = Sequence([meas]), Sequence([tr])
        frames.append(fr)
    ds.PerFrameFunctionalGroupsSequence = Sequence(frames)
    ds.PixelData = np.stack([(vol[:, :, k].T + 1024) for k in range(vol.shape[2])]).astype("<i2").tobytes()
    ds.save_as(str(f), enforce_file_format=True)


def test_enhanced_multiframe_with_all_geometry_per_frame(tmp_path):
    vol = _ramp((6, 5, 4))
    _write_multiframe(tmp_path / "mf.dcm", vol, 0)
    scan = read_scan(tmp_path / "mf.dcm")
    assert scan.data.shape == vol.shape and np.array_equal(scan.data, vol.astype(np.float32))
    assert np.allclose(scan.spacing_mm, [1.0, 1.0, 2.0])


def test_an_enhanced_series_split_over_several_files_is_one_volume(tmp_path):
    vol = _ramp((6, 5, 7))
    (tmp_path / "s").mkdir()
    _write_multiframe(tmp_path / "s" / "b.dcm", vol[:, :, 4:], 4)          # file order != slice order
    _write_multiframe(tmp_path / "s" / "a.dcm", vol[:, :, :4], 0)
    scan = read_scan(tmp_path / "s")
    assert scan.data.shape == vol.shape and np.array_equal(scan.data, vol.astype(np.float32))


def test_a_frame_missing_its_geometry_is_a_clean_error(tmp_path):
    _write_multiframe(tmp_path / "mf.dcm", _ramp((6, 5, 4)), 0, drop=2)
    with pytest.raises(ScanError, match="frame 3 .* ImagePositionPatient"):
        read_scan(tmp_path / "mf.dcm")


def test_summary_survives_an_import_with_no_wall_samples(tmp_path):
    rep = {"wall_error_mm": {"source_to_lumen_mm": {"n": 0}}, "edges": [], "entry_mm": [0, 0, 0],
           "vessels": {}}
    assert "no samples" in "\n".join(pipeline.summary_lines(rep, tmp_path))


def test_a_series_uid_wins_over_a_list_number(tmp_path):
    _write_series(tmp_path / "a", _ramp((6, 6, 4)), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1), 1.0, uid="2")
    _write_series(tmp_path / "b", _ramp((6, 6, 9)), (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1), 1.0, uid="1.9")
    assert read_scan(tmp_path, series="2").data.shape[2] == 4              # the UID, not the 2nd largest
    assert read_scan(tmp_path, series=1).data.shape[2] == 9
    with pytest.raises(ScanError, match="--list-series"):
        read_scan(tmp_path, series="7")


def test_a_failed_install_is_asked_about_again(lumen_home):
    seg = segmenters.TOTALSEGMENTATOR

    def broken(s, say):
        raise segmenters.SegmenterError("disk full")

    with pytest.raises(segmenters.SegmenterError, match="disk full"):
        segmenters.ensure(seg, why="w", ask=lambda _: "y", say=lambda *_: None, install=broken)
    assert seg.key not in segmenters._consents()
    segmenters.agree(seg, why="w", ask=lambda _: "y")
    job = segmenters.BackgroundInstall(seg, broken)
    with pytest.raises(segmenters.SegmenterError, match="disk full"):
        job.wait()
    asked = []
    with pytest.raises(segmenters.Declined):                            # asked again, and can say no
        segmenters.agree(seg, why="w", ask=lambda text: asked.append(text) or "n")
    assert len(asked) == 1


def test_the_question_goes_to_stderr_so_json_output_stays_clean(monkeypatch, capsys):
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    assert segmenters._ask_on_stderr("Install it now? ").strip() == "y"
    out = capsys.readouterr()
    assert out.out == "" and "Install it now?" in out.err
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with pytest.raises(EOFError):
        segmenters._ask_on_stderr("Install it now? ")


def test_linux_drives_that_cannot_hold_an_environment_are_recognised(tmp_path, monkeypatch):
    import pathlib
    import sys

    mounts = tmp_path / "mounts"
    mounts.write_text("/dev/sda1 / ext4 rw 0 0\n/dev/sdb1 /media/usb\\040drive exfat rw 0 0\n")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(segmenters, "_PROC_MOUNTS", mounts)
    monkeypatch.setattr("os.path.realpath", lambda p: str(p))
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)       # the mount points exist there
    assert segmenters._filesystem(pathlib.Path("/media/usb drive/lumen")) == "exfat"
    assert segmenters._filesystem(pathlib.Path("/home/me/.lumen")) == "ext4"
    assert "exfat" in segmenters._NO_ENV_FILESYSTEMS and "vfat" in segmenters._NO_ENV_FILESYSTEMS


def test_a_failed_import_leaves_no_copy_of_the_scan(tmp_path, lumen_home, monkeypatch):
    scan, labels, _ = _fork_scan()
    nib.save(nib.Nifti1Image(scan.data, np.diag([-1.0, -1.0, 1.0, 1.0]) @ scan.affine),
             str(tmp_path / "scan.nii.gz"))
    monkeypatch.setattr(region, "detect_region", lambda s: region.RegionReport(("body",), True, "picture"))
    monkeypatch.setattr(segmenters, "installed", lambda seg: True)

    def run_body(s, work, say=print):
        work.mkdir(parents=True)
        (work / "scan.nii.gz").write_bytes(b"copy of the scan")
        raise segmenters.SegmenterError("found nothing")

    monkeypatch.setattr(segmenters, "run_body", run_body)
    with pytest.raises(segmenters.SegmenterError, match="could not find the arteries"):
        pipeline.import_scan(tmp_path / "scan.nii.gz", tmp_path / "out", say=lambda *_: None)
    assert not (tmp_path / "out" / "work").exists()


def test_an_mr_saved_as_nifti_gets_the_mr_body_task(tmp_path, lumen_home, monkeypatch):
    from lumen.scan.io import is_ct

    mr = ScanVolume(np.full((8, 8, 8), 300.0, np.float32), np.eye(4), modality="")
    ct = ScanVolume(np.full((8, 8, 8), -1000.0, np.float32), np.eye(4), modality="")
    assert not is_ct(mr) and is_ct(ct)
    ran = []
    monkeypatch.setattr(segmenters, "_device", lambda seg: "cpu")
    monkeypatch.setattr(segmenters, "_class_map", lambda seg, task: {1: "aorta"})

    def fake_run(cmd, log, env=None, what="", timeout=0):
        ran.append([str(c) for c in cmd])
        raise segmenters.SegmenterError("stop here")

    monkeypatch.setattr(segmenters, "_run", fake_run)
    with pytest.raises(segmenters.SegmenterError):
        segmenters.run_body(mr, tmp_path / "w", say=lambda *_: None)
    assert ran[0][ran[0].index("--task") + 1] == "total_mr"


def test_an_install_that_fails_is_told_apart_from_finding_nothing(tmp_path, lumen_home, monkeypatch):
    scan, labels, _ = _fork_scan()
    nib.save(nib.Nifti1Image(scan.data, np.diag([-1.0, -1.0, 1.0, 1.0]) @ scan.affine),
             str(tmp_path / "scan.nii.gz"))

    def install(seg, say):
        if seg is segmenters.RSNA_INTRACRANIAL:
            raise segmenters.SegmenterError("download refused")
        _fake_install([])(seg, say)

    monkeypatch.setattr(segmenters, "_install", install)
    monkeypatch.setattr(segmenters, "run_body", lambda s, work, say=print: labels)
    monkeypatch.setattr(region, "detect_region",
                        lambda s: region.RegionReport(("head", "body"), True, "picture"))
    rep = pipeline.import_scan(tmp_path / "scan.nii.gz", tmp_path / "out", ask=lambda _: "y",
                               say=lambda *_: None)
    note = rep["segmentation"]["notes"][0]
    assert "could not be installed" in note and "run the import again" in note and "download refused" in note
    assert "rsna-intracranial" not in segmenters._consents()                 # asked again next time
    assert "totalsegmentator" in segmenters._consents()


def test_a_narrow_neck_below_the_skull_is_not_taken_into_the_head(monkeypatch):
    # slices feet to head: 40 trunk, 20 neck (head-sized area but 120 mm wide), 30 skull
    n_trunk, n_neck, n_head = 40, 20, 30
    a = np.r_[np.full(n_trunk, 600.0), np.full(n_neck, 140.0), np.full(n_head, 180.0)]
    w = np.r_[np.full(n_trunk, 330.0), np.full(n_neck, 120.0), np.full(n_head, 170.0)]
    b = np.r_[np.full(n_trunk, 40.0), np.full(n_neck, 10.0), np.full(n_head, 40.0)]
    m = {"area_cm2": a, "bone_cm2": b, "width_mm": w, "cut_off": np.zeros(len(a), bool),
         "step_mm": 2.5, "ct": True, "z0_mm": 0.0}
    monkeypatch.setattr(region, "_slice_measures", lambda v: m)
    scan = ScanVolume(np.zeros((2, 2, 2), np.float32), np.eye(4), modality="CT")
    r = region.detect_region(scan)
    assert r.regions == ("head", "body")
    # the head starts at the skull (less the 20 mm skull-base margin), not down the neck
    assert r.detail["head_from_mm"] == pytest.approx((n_trunk + n_neck) * 2.5 - 20.0)


def test_a_body_model_without_the_wanted_arteries_stops_before_running(tmp_path, lumen_home, monkeypatch):
    ran = []
    monkeypatch.setattr(segmenters, "_device", lambda seg: "cpu")
    monkeypatch.setattr(segmenters, "_class_map", lambda seg, task: {1: "liver", 2: "spleen"})
    monkeypatch.setattr(segmenters, "_run", lambda *a, **k: ran.append(a))
    ct = ScanVolume(np.full((8, 8, 8), -1000.0, np.float32), np.eye(4), modality="CT")
    with pytest.raises(segmenters.SegmenterError, match="none of the arteries"):
        segmenters.run_body(ct, tmp_path / "w", say=lambda *_: None)
    assert ran == []
