"""L2.0 — episode schema round-trip, validation, and firewall coverage (no Newton)."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from lumen.data import Episode, EpisodeMeta, Outcome, Step, validate


def _episode(n=4, provenance="procedural"):
    return Episode(
        meta=EpisodeMeta(asset_ref="straight.json", dt=5e-3, provenance=provenance,
                         notes={"true_C10": 4000.0}),
        steps=[Step(t=i * 5e-3, action={"insertion": 1.0},
                    kinematics={"tip_mm": [0.0, 0.0, float(i)], "tip_s": float(i),
                                "node_positions_ref": f"{i:03d}_nodes.npy"},
                    obs_modality="fluoro", obs_ref=f"{i:03d}.npy",
                    obs=np.full((4, 4), float(i)),
                    node_positions=np.full((3, 3), float(i)))
               for i in range(n)],
        outcome=Outcome(success=True, final_dist=0.4, steps=n, label="straight"))


def test_round_trip_manifest_and_sidecars(tmp_path):
    ep = _episode()
    ep.save(tmp_path)
    back = Episode.load(tmp_path)
    norm = lambda m: json.dumps(m, sort_keys=True)
    assert norm(back.manifest()) == norm(ep.manifest())          # scalars round-trip
    assert np.array_equal(back.steps[2].load_obs(tmp_path), np.full((4, 4), 2.0))
    assert np.array_equal(back.steps[1].load_nodes(tmp_path), np.full((3, 3), 1.0))
    assert (tmp_path / "manifest.json").exists() and (tmp_path / "obs" / "000.npy").exists()


def test_lazy_obs_not_loaded_on_manifest_load(tmp_path):
    ep = _episode()
    ep.save(tmp_path)
    back = Episode.load(tmp_path)
    assert back.steps[0].obs is None                             # lazy: manifest load leaves arrays unread
    assert back.steps[0].obs_ref == "000.npy"


def test_validate_rejects_malformed():
    validate(_episode())                                         # the good one passes
    with pytest.raises(ValueError):
        validate(Episode(steps=[]))                             # no steps
    bad_t = _episode(); bad_t.steps[2].t = 0.0                  # time goes backwards
    with pytest.raises(ValueError):
        validate(bad_t)
    nan_tip = _episode(); nan_tip.steps[1].kinematics["tip_mm"] = [0.0, np.nan, 1.0]
    with pytest.raises(ValueError):
        validate(nan_tip)
    bad_mod = _episode(); bad_mod.steps[0].obs_modality = "ultrasound"
    with pytest.raises(ValueError):
        validate(bad_mod)
    with pytest.raises(ValueError):
        validate(Episode(meta=EpisodeMeta(provenance="leaked"), steps=[Step()], outcome=Outcome()))


def test_patient_provenance_is_valid_but_version_pinned():
    validate(Episode(meta=EpisodeMeta(provenance="patient(private)"), steps=[Step()],
                     outcome=Outcome()))                         # valid value; firewall (not validate) blocks commit
    with pytest.raises(ValueError):
        validate(Episode(meta=EpisodeMeta(version="lumen-episode/99"), steps=[Step()],
                         outcome=Outcome()))


def _load_firewall():
    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "check_firewall.py"
    spec = importlib.util.spec_from_file_location("check_firewall", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_firewall_flags_patient_manifest(tmp_path, monkeypatch):
    # episode manifests are auto-covered by the firewall (it scans every *.json for a
    # top-level provenance key). Point the checker at a tree with both kinds and assert
    # it flags ONLY the patient one — pinning that provenance sits where the firewall looks.
    _episode(provenance="procedural").save(tmp_path / "ok")
    _episode(provenance="patient(private)").save(tmp_path / "leak")
    fw = _load_firewall()
    monkeypatch.setattr(fw, "ROOT", tmp_path)
    problems = fw.check_provenance()
    assert any("leak" in p for p in problems)                    # patient manifest flagged
    assert not any("ok" in p for p in problems)                  # procedural manifest clean
