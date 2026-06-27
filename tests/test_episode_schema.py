"""L2.0 — episode schema round-trip, validation, and firewall coverage (no Newton)."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from lumen.assets import procedural
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
        outcome=Outcome(success=True, final_dist=0.4, steps=n, label="straight"),
        asset=procedural.straight_tube(80.0, 2.0))


def test_round_trip_manifest_and_sidecars(tmp_path):
    ep = _episode()
    ep.save(tmp_path)
    back = Episode.load(tmp_path)
    norm = lambda m: json.dumps(m, sort_keys=True)
    assert norm(back.manifest()) == norm(ep.manifest())          # scalars round-trip
    assert np.array_equal(back.steps[2].load_obs(tmp_path), np.full((4, 4), 2.0))
    assert np.array_equal(back.steps[1].load_nodes(tmp_path), np.full((3, 3), 1.0))
    assert (tmp_path / "manifest.json").exists() and (tmp_path / "obs" / "000.npy").exists()
    assert back.load_asset(tmp_path).edges[0].id == "e0"
    assert (tmp_path / "straight.json").exists()


def test_load_rejects_disagreeing_toplevel_mirror(tmp_path):
    # the top-level version/provenance mirror is a checksum on the canonical meta.*;
    # a hand-edited manifest where they disagree must fail loud (trust boundary — a
    # top-level "procedural" must not be able to hide a patient meta).
    _episode().save(tmp_path)
    man = json.loads((tmp_path / "manifest.json").read_text())
    man["provenance"] = "patient(private)"                       # top-level now lies vs meta
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="disagrees with"):
        Episode.load(tmp_path)


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


def test_validate_rejects_non_mapping_metadata_fields():
    bad = _episode()
    bad.meta.calibration = []
    with pytest.raises(ValueError, match="meta.calibration must be a mapping"):
        validate(bad)

    bad = _episode()
    bad.outcome.metrics = []
    with pytest.raises(ValueError, match="outcome.metrics must be a mapping"):
        validate(bad)


def test_patient_provenance_is_valid_but_version_pinned():
    validate(Episode(meta=EpisodeMeta(provenance="patient(private)"), steps=[Step()],
                     outcome=Outcome(steps=1)))                  # valid value; firewall (not validate) blocks commit
    with pytest.raises(ValueError):
        validate(Episode(meta=EpisodeMeta(version="lumen-episode/99"), steps=[Step()],
                         outcome=Outcome(steps=1)))


def test_validate_rejects_unsafe_and_inconsistent_refs():
    dup = _episode(2); dup.steps[1].obs_ref = dup.steps[0].obs_ref       # H1: clobber
    with pytest.raises(ValueError, match="duplicate obs_ref"):
        validate(dup)
    dup_n = _episode(2); dup_n.steps[1].kinematics["node_positions_ref"] = \
        dup_n.steps[0].kinematics["node_positions_ref"]
    with pytest.raises(ValueError, match="duplicate node_positions_ref"):
        validate(dup_n)
    for bad_ref in ("../evil.npy", "a/b.npy", "/etc/x.npy"):             # H2: traversal
        ev = _episode(1); ev.steps[0].obs_ref = bad_ref
        with pytest.raises(ValueError, match="bare filename"):
            validate(ev)


def test_validate_modality_ref_consistency():
    drop = _episode(1); drop.steps[0].obs_ref = None                     # M1: fluoro needs a ref
    with pytest.raises(ValueError, match="requires obs_ref"):
        validate(drop)
    spurious = _episode(1); spurious.steps[0].obs_modality = "none"      # none must not carry one
    with pytest.raises(ValueError, match="obs_modality='none'"):
        validate(spurious)


def test_validate_outcome_and_tip_shape():
    mism = _episode(3); mism.outcome.steps = 99                          # L1: count drift
    with pytest.raises(ValueError, match="outcome.steps"):
        validate(mism)
    short = _episode(1); short.steps[0].kinematics["tip_mm"] = [0.0, 1.0]  # L2: not a 3-vector
    with pytest.raises(ValueError, match="length-3"):
        validate(short)
    junk = _episode(1); junk.steps[0].kinematics["tip_mm"] = ["x", "y", "z"]  # numeric contract
    with pytest.raises(ValueError, match="not numeric"):
        validate(junk)


def test_validate_root_mode_checks_files_exist(tmp_path):
    ep = _episode(2)
    ep.save(tmp_path)
    validate(ep, root=tmp_path)                                          # all sidecars present
    (tmp_path / "obs" / "001.npy").unlink()                              # delete one
    with pytest.raises(ValueError, match="missing on disk"):
        validate(Episode.load(tmp_path), root=tmp_path)


def test_validate_root_mode_checks_annotation_sidecars(tmp_path):
    ep = _episode(1)
    ep.steps[0].annotations = {"device_mask_ref": "000_device_mask.npy"}
    ep.steps[0].annotation_arrays = {"device_mask": np.ones((4, 4), dtype=np.uint8)}
    ep.save(tmp_path)
    assert np.array_equal(Episode.load(tmp_path).steps[0].load_annotation(tmp_path, "device_mask"),
                          np.ones((4, 4), dtype=np.uint8))

    (tmp_path / "obs" / "000_device_mask.npy").unlink()
    with pytest.raises(ValueError, match="annotation sidecar missing"):
        validate(Episode.load(tmp_path), root=tmp_path)


def test_validate_root_mode_checks_device_mask_shape_and_dtype(tmp_path):
    ep = _episode(1)
    ep.steps[0].annotations = {"device_mask_ref": "000_device_mask.npy"}
    ep.steps[0].annotation_arrays = {"device_mask": np.ones((4, 4), dtype=np.uint8)}
    ep.save(tmp_path)

    np.save(tmp_path / "obs" / "000_device_mask.npy", np.ones((3, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="device_mask shape"):
        validate(Episode.load(tmp_path), root=tmp_path)

    np.save(tmp_path / "obs" / "000_device_mask.npy", np.ones((4, 4), dtype=float))
    with pytest.raises(ValueError, match="bool/unsigned integer"):
        validate(Episode.load(tmp_path), root=tmp_path)

    np.save(tmp_path / "obs" / "000_device_mask.npy", np.ones((4, 4, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="must be 2-D"):
        validate(Episode.load(tmp_path), root=tmp_path)


def test_save_rejects_bad_in_memory_device_mask(tmp_path):
    ep = _episode(1)
    ep.steps[0].annotations = {"device_mask_ref": "000_device_mask.npy"}
    ep.steps[0].annotation_arrays = {"device_mask": np.ones((3, 4), dtype=np.uint8)}

    with pytest.raises(ValueError, match="device_mask shape"):
        ep.save(tmp_path)


def test_validate_checks_keypoint_metadata_and_bounds(tmp_path):
    good = _episode(1)
    good.steps[0].annotations = {
        "keypoints": {
            "tip": {"uv": [1.0, 2.0], "present": True},
            "base": {"present": False},
        }
    }
    good.save(tmp_path / "good")
    validate(Episode.load(tmp_path / "good"), root=tmp_path / "good")

    bad_uv = _episode(1)
    bad_uv.steps[0].annotations = {"keypoints": {"tip": {"uv": [1.0], "present": True}}}
    with pytest.raises(ValueError, match="uv must be length-2"):
        bad_uv.save(tmp_path / "bad_uv")

    missing_uv = _episode(1)
    missing_uv.steps[0].annotations = {"keypoints": {"tip": {"present": True}}}
    with pytest.raises(ValueError, match="missing uv"):
        missing_uv.save(tmp_path / "missing_uv")

    out_of_bounds = _episode(1)
    out_of_bounds.steps[0].annotations = {
        "keypoints": {"tip": {"uv": [1.0, 0.0], "present": True}}
    }
    out_of_bounds.save(tmp_path / "out_of_bounds")
    man = json.loads((tmp_path / "out_of_bounds" / "manifest.json").read_text())
    man["steps"][0]["annotations"]["keypoints"]["tip"]["uv"] = [4.0, 0.0]
    (tmp_path / "out_of_bounds" / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="outside observation shape"):
        validate(Episode.load(tmp_path / "out_of_bounds"), root=tmp_path / "out_of_bounds")


def test_validate_rejects_cross_type_sidecar_clobbering():
    ep = _episode(1)
    ep.steps[0].annotations = {"device_mask_ref": ep.steps[0].obs_ref}

    with pytest.raises(ValueError, match="duplicate sidecar refs"):
        validate(ep)


def test_validate_root_mode_checks_local_asset_exists(tmp_path):
    ep = _episode(1)
    ep.save(tmp_path)
    (tmp_path / ep.meta.asset_ref).unlink()
    with pytest.raises(ValueError, match="asset_ref sidecar missing"):
        validate(Episode.load(tmp_path), root=tmp_path)


def test_save_rejects_loaded_episode_with_dangling_local_asset_ref(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _episode(1).save(src)

    loaded = Episode.load(src)

    with pytest.raises(ValueError, match="local asset_ref requires ep.asset"):
        loaded.save(dst)


def test_save_validates_and_guards_data_loss(tmp_path):
    bad = _episode(2); bad.steps[1].t = -1.0                             # backwards time
    with pytest.raises(ValueError):
        bad.save(tmp_path / "a")                                         # save is the gate
    lossy = _episode(1); lossy.steps[0].obs_modality = "none"            # obs set, ref dropped
    lossy.steps[0].obs_ref = None
    with pytest.raises(ValueError, match="obs set but obs_ref missing"):
        lossy.save(tmp_path / "b")


def test_load_obs_rejects_traversal(tmp_path):
    ep = _episode(1)
    ep.save(tmp_path)
    back = Episode.load(tmp_path)
    back.steps[0].obs_ref = "../../escape.npy"                           # tampered manifest
    with pytest.raises(ValueError, match="escapes the obs directory"):
        back.steps[0].load_obs(tmp_path)


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
    # a hand-edited file that hides provenance ONLY in a nested object must still be caught
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "manifest.json").write_text(
        json.dumps({"meta": {"provenance": "patient(private)"}}))      # no top-level key
    fw = _load_firewall()
    monkeypatch.setattr(fw, "ROOT", tmp_path)
    problems = fw.check_provenance()
    assert any("leak" in p for p in problems)                    # patient manifest flagged
    assert any("nested" in p for p in problems)                  # nested-only provenance also flagged
    assert not any("ok" in p for p in problems)                  # procedural manifest clean
