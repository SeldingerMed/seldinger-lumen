"""The open-core firewall: patient-derived geometry only in openly licensed shipped cases."""

from __future__ import annotations

import gzip
import importlib.util
import json
import pathlib
import shutil

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def firewall(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("check_firewall", ROOT / "tools" / "check_firewall.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "lumen" / "cases").mkdir(parents=True)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "CASES", tmp_path / "lumen" / "cases")
    return mod, tmp_path


def _case(root, cid, **card):
    d = root / "lumen" / "cases" / cid
    d.mkdir(parents=True)
    base = {"licence": "CC BY 4.0", "citation": "A et al. 2025", "source_url": "https://example.org"}
    (d / "case.json").write_text(json.dumps({**base, **card}))
    (d / "asset.json.gz").write_bytes(gzip.compress(json.dumps({"provenance": "imported(scan)"}).encode()))
    return d


def test_the_repository_passes():
    spec = importlib.util.spec_from_file_location("check_firewall", ROOT / "tools" / "check_firewall.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0


def test_an_openly_licensed_case_is_allowed(firewall):
    mod, root = firewall
    _case(root, "ok", licence="CC0")
    assert mod.check_provenance() == []


@pytest.mark.parametrize("card", [{"licence": "CC BY-NC-SA 4.0"}, {"licence": None},
                                  {"citation": ""}, {"source_url": ""}])
def test_a_case_without_an_open_licence_or_credit_is_refused(firewall, card):
    mod, root = firewall
    _case(root, "bad", **card)
    assert len(mod.check_provenance()) == 1


def test_patient_geometry_outside_the_cases_is_refused_even_gzipped(firewall):
    mod, root = firewall
    d = _case(root, "ok")
    (root / "data").mkdir()
    shutil.copy(d / "asset.json.gz", root / "data" / "scan.json.gz")
    (root / "data" / "scan.json").write_text(json.dumps({"meta": {"provenance": "imported(scan)"}}))
    bad = mod.check_provenance()
    assert len(bad) == 2 and all("data" in b for b in bad)


def test_a_case_missing_its_card_is_refused(firewall):
    mod, root = firewall
    d = _case(root, "nocard")
    (d / "case.json").unlink()
    assert len(mod.check_provenance()) == 1


def test_private_data_is_refused_even_inside_an_open_case(firewall):
    mod, root = firewall
    d = _case(root, "ok")
    (d / "asset.json.gz").write_bytes(gzip.compress(json.dumps({"provenance": "patient(private)"}).encode()))
    assert len(mod.check_provenance()) == 1


def test_a_private_marker_behind_an_approved_one_is_still_refused(firewall):
    mod, root = firewall
    d = _case(root, "ok")
    doc = {"provenance": "imported(scan)", "parts": [{"provenance": "patient(private)"}]}
    (d / "asset.json.gz").write_bytes(gzip.compress(json.dumps(doc).encode()))
    assert len(mod.check_provenance()) == 1


def test_a_case_asset_must_carry_its_provenance_marker(firewall):
    mod, root = firewall
    d = _case(root, "ok")
    (d / "asset.json.gz").write_bytes(gzip.compress(json.dumps({"edges": []}).encode()))
    assert len(mod.check_provenance()) == 1


def test_an_unreadable_case_asset_is_refused(firewall):
    mod, root = firewall
    d = _case(root, "ok")
    (d / "asset.json.gz").write_bytes(b"not gzip at all")
    assert len(mod.check_provenance()) == 1


def test_build_output_is_not_mistaken_for_loose_geometry(firewall):
    mod, root = firewall
    out = root / "build" / "lib" / "lumen" / "cases" / "ok"
    out.mkdir(parents=True)
    (out / "surface.npz").write_bytes(b"x")
    assert mod.check_geometry_files() == []


def test_only_a_cases_own_files_get_its_approval(firewall):
    mod, root = firewall
    d = _case(root, "ok")
    (d / "episode.json").write_text(json.dumps({"provenance": "imported(scan)"}))
    (d / "extra.stl").write_bytes(b"solid x\nendsolid x\n")
    assert len(mod.check_provenance()) == 1 and len(mod.check_geometry_files()) == 1


def test_surface_files_outside_the_cases_are_refused(firewall):
    mod, root = firewall
    (root / "data").mkdir()
    for name in ("vessel.npz", "vessel.stl", "scan.nii.gz", "scan.nrrd"):
        (root / "data" / name).write_bytes(b"x")
    d = _case(root, "ok")
    (d / "surface.npz").write_bytes(b"x")                   # a case's own surface is fine
    bad = mod.check_geometry_files()
    assert len(bad) == 4 and all("data" in b for b in bad)
