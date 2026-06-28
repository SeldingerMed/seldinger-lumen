"""L2.2 — corpus iteration + replay (pure numpy; episodes built via the schema)."""

import numpy as np
import pytest

from lumen.assets import procedural
from lumen.data import (Episode, EpisodeDataset, EpisodeMeta, Outcome, Step,
                        annotation_coverage, replay, summarize)


def _ep(label, n=3, success=True, modality="fluoro"):
    return Episode(
        meta=EpisodeMeta(asset_ref=f"{label}.json"),
        steps=[Step(t=i * 0.1, action={"insertion": 1.0},
                    kinematics={"tip_mm": [0.0, 0.0, float(i)], "tip_s": float(i)},
                    obs_modality=modality, obs_ref=(f"{i:03d}.npy" if modality != "none" else None),
                    obs=(np.full((3, 3), float(i)) if modality != "none" else None))
               for i in range(n)],
        outcome=Outcome(success=success, final_dist=0.4 if success else 9.0, steps=n, label=label),
        asset=procedural.straight_tube(80.0, 2.0))


def _corpus(tmp_path):
    _ep("straight", 3, True).save(tmp_path / "a")
    _ep("stenosis", 5, False).save(tmp_path / "b")
    return EpisodeDataset(tmp_path)


def test_discovers_and_indexes_episodes(tmp_path):
    ds = _corpus(tmp_path)
    assert len(ds) == 2
    labels = {ds[i].outcome.label for i in range(len(ds))}
    assert labels == {"straight", "stenosis"}
    assert all(hasattr(ep, "root") for ep in ds)            # runtime root attached for lazy obs


def test_replay_yields_steps_with_lazy_obs(tmp_path):
    ds = _corpus(tmp_path)
    ep = next(e for e in ds if e.outcome.label == "straight")
    steps = list(replay(ep))
    assert len(steps) == 3
    t, action, kin, obs = steps[2]
    assert t == pytest.approx(0.2) and action["insertion"] == 1.0
    assert kin["tip_s"] == 2.0
    assert obs.shape == (3, 3) and np.array_equal(obs, np.full((3, 3), 2.0))   # loaded on demand


def test_replay_can_yield_lazy_annotations_with_obs(tmp_path):
    ep = _ep("seg", 2, True)
    ep.steps[1].annotations = {
        "device_mask_ref": "001_device_mask.npy",
        "keypoints": {"tip": {"uv": [1.0, 2.0], "present": True}},
    }
    ep.steps[1].annotation_arrays = {"device_mask": np.ones((3, 3), dtype=np.uint8)}
    ep.save(tmp_path / "seg")
    ds = EpisodeDataset(tmp_path)

    sample = list(replay(ds[0], include_annotations=True))[1]
    t, action, kin, obs, annotations = sample

    assert t == pytest.approx(0.1)
    assert action["insertion"] == 1.0
    assert kin["tip_s"] == 1.0
    assert np.array_equal(obs, np.full((3, 3), 1.0))
    assert np.array_equal(annotations["device_mask"], np.ones((3, 3), dtype=np.uint8))
    assert annotations["device_mask_ref"] == "001_device_mask.npy"
    assert annotations["keypoints"]["tip"]["present"] is True
    annotations["device_mask_ref"] = "mutated.npy"
    annotations["keypoints"]["tip"]["present"] = False
    assert ds[0].steps[1].annotations["device_mask_ref"] == "001_device_mask.npy"
    assert ds[0].steps[1].annotations["keypoints"]["tip"]["present"] is True


def test_replay_none_modality_has_no_obs(tmp_path):
    _ep("plain", 2, True, modality="none").save(tmp_path / "p")
    ds = EpisodeDataset(tmp_path)
    assert all(obs is None for _, _, _, obs in replay(ds[0]))


def test_validate_on_load_catches_missing_sidecar(tmp_path):
    _ep("straight", 3, True).save(tmp_path / "a")
    (tmp_path / "a" / "obs" / "001.npy").unlink()            # corrupt the corpus
    with pytest.raises(ValueError, match="missing on disk"):
        EpisodeDataset(tmp_path)[0]                          # fail fast on access
    lenient = EpisodeDataset(tmp_path, validate_on_load=False)[0]   # opt-out for repair
    assert len(lenient.steps) == 3


def test_summarize_corpus(tmp_path):
    s = summarize(_corpus(tmp_path))
    assert s["episodes"] == 2
    assert s["success_rate"] == pytest.approx(0.5)
    assert s["mean_steps"] == pytest.approx(4.0)            # (3 + 5) / 2
    assert s["labels"] == {"straight": 1, "stenosis": 1}
    assert s["total_steps"] == 8
    assert s["modalities"] == {"fluoro": 8}


def test_annotation_coverage_is_manifest_only_for_cv_readiness(tmp_path):
    ep = _ep("seg", 3, True)
    for i, step in enumerate(ep.steps[:2]):
        step.annotations = {
            "device_mask_ref": f"{i:03d}_device_mask.npy",
            "keypoints": {
                "base": {"uv": [0.0, 0.0], "present": i == 0},
                "tip": {"uv": [1.0, 2.0], "present": True},
                "nodes": [
                    {"uv": [0.0, 0.0], "present": True},
                    {"present": False},
                ],
            },
        }
    ep.save(tmp_path / "seg")
    back = Episode.load(tmp_path / "seg")

    cov = annotation_coverage(back)

    assert cov == {
        "steps": 3,
        "modalities": {"fluoro": 3},
        "annotation_steps": 2,
        "sidecars": {"device_mask": 2},
        "keypoint_steps": 2,
        "keypoints_present": {"base": 1, "tip": 2, "nodes": 2},
        "keypoints_total": {"base": 2, "tip": 2, "nodes": 4},
    }
    s = summarize(EpisodeDataset(tmp_path, validate_on_load=False))
    assert s["annotations"] == {"device_mask": 2}
    assert s["annotation_steps"] == 2
    assert s["keypoint_steps"] == 2
    assert s["keypoints_present"] == {"base": 1, "tip": 2, "nodes": 2}
    assert s["keypoints_total"] == {"base": 2, "tip": 2, "nodes": 4}


def test_empty_corpus(tmp_path):
    ds = EpisodeDataset(tmp_path)
    assert len(ds) == 0 and list(ds) == []
    assert summarize(ds)["episodes"] == 0                   # no crash on an empty corpus


def test_discovers_nested_and_obs_named_dirs(tmp_path):
    _ep("deep", 2).save(tmp_path / "x" / "y" / "deep")     # nested discovery
    _ep("obs", 2).save(tmp_path / "obs")                   # a dir literally named "obs" (H1)
    ds = EpisodeDataset(tmp_path)
    assert len(ds) == 2
    assert {ep.outcome.label for ep in ds} == {"deep", "obs"}


def test_corrupt_manifest_raises_path_tagged_valueerror(tmp_path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "manifest.json").write_text('{"not": "an episode"}')
    ds = EpisodeDataset(tmp_path)
    with pytest.raises(ValueError, match="failed to load/validate"):
        ds[0]                                              # KeyError -> clear path-tagged ValueError


def test_slice_indexing_returns_list(tmp_path):
    for k in range(3):
        _ep(f"c{k}", 2).save(tmp_path / f"c{k}")
    ds = EpisodeDataset(tmp_path)
    sl = ds[:2]
    assert isinstance(sl, list) and len(sl) == 2 and all(isinstance(e, Episode) for e in sl)


def test_replay_dicts_are_copies(tmp_path):
    ds = _corpus(tmp_path)
    ep = ds[0]
    a, b = list(replay(ep)), list(replay(ep))
    assert a[0][1] is not b[0][1]                          # M4: action dicts are distinct objects
    a[0][2]["tip_s"] = -999.0                              # mutating a replay must not corrupt the next
    assert b[0][2]["tip_s"] != -999.0


def test_summarize_does_not_read_sidecars(tmp_path):
    _ep("straight", 3, True).save(tmp_path / "a")
    (tmp_path / "a" / "obs" / "001.npy").unlink()          # delete a sidecar
    # validate_on_load would raise here, but summarize is manifest-only -> still works
    assert summarize(EpisodeDataset(tmp_path))["episodes"] == 1


def test_nonexistent_root_warns(tmp_path):
    with pytest.warns(UserWarning, match="not a directory"):
        EpisodeDataset(tmp_path / "nope")


def test_replay_corpus_example_handles_missing_root_without_warning(tmp_path, capsys):
    import warnings

    from examples.replay_corpus import main

    missing = tmp_path / "nope"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        main(str(missing))

    out = capsys.readouterr().out
    assert "run `lumen capture" in out
    assert seen == []


def test_replay_corpus_example_prints_clinical_endpoint_flags(tmp_path, capsys):
    from examples.replay_corpus import main
    from lumen.sensors.carm import CArm

    ep = _ep("clinical", 2, True)
    carm = CArm.looking_at([0.0, 0.0, 40.0], axis=(1.0, 0.0, 0.0), nu=8, nv=8)
    ep.meta.device = {"guidewire": {"radius": 0.2}}
    ep.meta.sensor = {"modality": "fluoro", "nu": 8, "nv": 8}
    ep.meta.calibration = {"type": "carm", "views": [carm.to_dict()]}
    ep.meta.labels = {"procedure": "navigation"}
    for i, step in enumerate(ep.steps):
        step.annotations = {
            "device_mask_ref": f"{i:03d}_device_mask.npy",
            "keypoints": {
                "base": {"uv": [0.0, 0.0], "present": i == 0},
                "tip": {"uv": [1.0, 2.0], "present": True},
            },
        }
        step.annotation_arrays = {"device_mask": np.ones((3, 3), dtype=np.uint8)}
    ep.outcome.metrics = {
        "tip_target": {"success": True},
        "wall_safety": {"perforation_risk": False},
        "branch_choice": {"correct": True},
    }
    ep.save(tmp_path / "clinical")

    main(str(tmp_path))

    out = capsys.readouterr().out
    assert "tip_target=True" in out
    assert "wall_risk=False" in out
    assert "branch=True" in out
    assert "device_mask=2/2" in out
    assert "keypoints(base=1/2 tip=2/2)" in out


def test_replay_corpus_example_skips_invalid_bundles_but_lists_valid_ones(tmp_path, capsys):
    from examples.replay_corpus import main
    from lumen.sensors.carm import CArm

    good = _ep("valid", 2, True)
    carm = CArm.looking_at([0.0, 0.0, 40.0], axis=(1.0, 0.0, 0.0), nu=8, nv=8)
    good.meta.device = {"guidewire": {"radius": 0.2}}
    good.meta.sensor = {"modality": "fluoro", "nu": 8, "nv": 8}
    good.meta.calibration = {"type": "carm", "views": [carm.to_dict()]}
    good.meta.labels = {"procedure": "navigation"}
    good.save(tmp_path / "valid")
    _ep("loose", 2, True).save(tmp_path / "loose")       # episode-valid, not bundle-valid

    main(str(tmp_path))

    out = capsys.readouterr().out
    assert "valid" in out
    assert "skipped invalid bundles" in out
    assert "loose" in out


def test_summarize_segregates_probe_from_navigation(tmp_path):
    _ep("straight", 3, True).save(tmp_path / "nav")        # navigation (no kind -> default)
    probe = _ep("wall_probe", 2, True)                     # a wall-probe masquerading as success
    probe.meta.notes["episode_kind"] = "wall_probe"
    probe.save(tmp_path / "probe")
    s = summarize(EpisodeDataset(tmp_path))
    assert s["episodes"] == 2 and s["navigation"] == 1     # only the nav episode is "navigation"
    assert s["kinds"] == {"navigation": 1, "wall_probe": 1}
    assert s["success_rate"] == 1.0                        # probe's success doesn't inflate it (over nav only)
