"""Self-contained case bundle contract.

A case bundle is stricter than a loose Episode: it must carry every artifact a
CV/endo consumer needs to replay the case from one directory.
"""

import numpy as np
import pytest

from lumen.assets import procedural
from lumen.data import CaseBundle, Episode, EpisodeMeta, Outcome, Step
from lumen.sensors.carm import CArm


def _bundle_episode():
    asset = procedural.straight_tube(80.0, 2.0)
    carm = CArm.looking_at([0.0, 0.0, 40.0], axis=(1.0, 0.0, 0.0),
                           distance=120.0, sdd=240.0, width=64.0, height=64.0,
                           nu=16, nv=16)
    return Episode(
        meta=EpisodeMeta(
            asset_ref="asset.json",
            dt=0.1,
            device={"guidewire": {"radius": 0.2, "n_nodes": 3, "node_spacing": 2.0}},
            sensor={"modality": "fluoro", "nu": 16, "nv": 16},
            calibration={"type": "carm", "views": [carm.to_dict()]},
            labels={"procedure": "navigation", "anatomy": "straight_tube"},
        ),
        steps=[
            Step(t=0.0, action={"insertion": 1.0, "twist": 0.0},
                 kinematics={"tip_mm": [0.0, 0.0, 2.0], "tip_s": 2.0,
                             "node_positions_ref": "000_nodes.npy"},
                 obs_modality="fluoro", obs_ref="000.npy",
                 obs=np.ones((16, 16)),
                 node_positions=np.zeros((3, 3))),
            Step(t=0.1, action={"insertion": 0.5, "twist": 0.1},
                 kinematics={"tip_mm": [0.0, 0.0, 4.0], "tip_s": 4.0,
                             "node_positions_ref": "001_nodes.npy"},
                 obs_modality="fluoro", obs_ref="001.npy",
                 obs=np.full((16, 16), 2.0),
                 node_positions=np.ones((3, 3))),
        ],
        outcome=Outcome(success=True, final_dist=0.5, steps=2, label="straight_success"),
        asset=asset,
    )


def test_case_bundle_loads_every_replay_input_from_one_directory(tmp_path):
    ep = _bundle_episode()
    ep.save(tmp_path)

    bundle = CaseBundle.load(tmp_path)

    assert bundle.asset.edges[0].id == "e0"
    assert bundle.device_definitions["guidewire"]["radius"] == 0.2
    assert bundle.calibration["type"] == "carm"
    assert bundle.labels == {
        "outcome": "straight_success",
        "procedure": "navigation",
        "anatomy": "straight_tube",
    }

    replayed = list(bundle.replay())
    assert len(replayed) == 2
    assert replayed[1][1] == {"insertion": 0.5, "twist": 0.1}
    assert np.array_equal(replayed[1][3], np.full((16, 16), 2.0))
    assert np.array_equal(bundle.episode.steps[1].load_nodes(bundle.root), np.ones((3, 3)))


def test_case_bundle_keeps_meta_labels_and_intentional_empty_outcome_label(tmp_path):
    ep = _bundle_episode()
    ep.outcome.label = ""
    ep.save(tmp_path)

    bundle = CaseBundle.load(tmp_path)

    assert bundle.labels["outcome"] == ""
    assert bundle.labels["procedure"] == "navigation"
    assert bundle.labels["anatomy"] == "straight_tube"


def test_case_bundle_rejects_missing_required_bundle_artifacts(tmp_path):
    ep = _bundle_episode()
    ep.save(tmp_path / "ok")
    (tmp_path / "ok" / "asset.json").unlink()
    with pytest.raises(ValueError, match="asset"):
        CaseBundle.load(tmp_path / "ok")

    no_cal = _bundle_episode()
    no_cal.meta.calibration = {}
    no_cal.save(tmp_path / "no_cal")
    with pytest.raises(ValueError, match="calibration"):
        CaseBundle.load(tmp_path / "no_cal")

    external = _bundle_episode()
    external.meta.asset_ref = "s3://private/case.asset.json"
    external.asset = None
    external.save(tmp_path / "external")
    with pytest.raises(ValueError, match="local asset sidecar"):
        CaseBundle.load(tmp_path / "external")
