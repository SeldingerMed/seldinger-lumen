"""L2.2 — corpus iteration + replay (pure numpy; episodes built via the schema)."""

import numpy as np
import pytest

from lumen.data import (Episode, EpisodeDataset, EpisodeMeta, Outcome, Step, replay,
                        summarize)


def _ep(label, n=3, success=True, modality="fluoro"):
    return Episode(
        meta=EpisodeMeta(asset_ref=f"{label}.json"),
        steps=[Step(t=i * 0.1, action={"insertion": 1.0},
                    kinematics={"tip_mm": [0.0, 0.0, float(i)], "tip_s": float(i)},
                    obs_modality=modality, obs_ref=(f"{i:03d}.npy" if modality != "none" else None),
                    obs=(np.full((3, 3), float(i)) if modality != "none" else None))
               for i in range(n)],
        outcome=Outcome(success=success, final_dist=0.4 if success else 9.0, steps=n, label=label))


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


def test_empty_corpus(tmp_path):
    ds = EpisodeDataset(tmp_path)
    assert len(ds) == 0 and list(ds) == []
    assert summarize(ds)["episodes"] == 0                   # no crash on an empty corpus
