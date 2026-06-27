"""L2.1 — synthetic capture: drive a sim, log paired observations, save an episode.

Needs Newton+Warp (it runs the Layer-0 sim) → importorskip, like the other RL tests."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.assets import procedural                      # noqa: E402
from lumen.data import CaseBundle, Episode, EpisodeRecorder, rollout_episode, validate  # noqa: E402
from lumen.data.capture import SimDiverged              # noqa: E402
from lumen.sensors import FluoroSensor, LuminalCamera    # noqa: E402


def test_rollout_fluoro_episode_round_trips(tmp_path):
    asset = procedural.straight_tube(80.0, 2.0)
    ep = rollout_episode(asset, sensor=FluoroSensor(res=20, nu=24, nv=24, n_samples=40),
                         max_steps=6, asset_ref="straight.json", notes={"true_C10": 4000.0})
    validate(ep)
    assert ep.steps and ep.outcome.steps == len(ep.steps)
    # the tip advances down the vessel (monotone insertion)
    assert ep.steps[-1].kinematics["tip_s"] >= ep.steps[0].kinematics["tip_s"]
    # every step carries a paired fluoro frame + node positions
    assert all(s.obs_modality == "fluoro" and s.obs_ref for s in ep.steps)
    assert ep.meta.notes["true_C10"] == 4000.0          # sim2sim ground truth rides in notes
    assert ep.meta.notes["procedure"] == "endovascular_navigation"

    ep.save(tmp_path)
    back = Episode.load(tmp_path)
    validate(back, root=tmp_path)                        # all sidecars present
    bundle = CaseBundle.load(tmp_path)
    assert bundle.calibration["type"] == "carm"
    assert bundle.labels["procedure"] == "endovascular_navigation"
    assert (tmp_path / "straight.json").exists()
    assert back.load_asset(tmp_path).edges[0].centerline_mm
    assert back.steps[0].load_obs(tmp_path).shape == (24, 24)
    assert back.steps[0].load_nodes(tmp_path).shape[1] == 3
    assert back.steps[0].load_annotation(tmp_path, "device_mask").shape == (24, 24)
    assert back.steps[0].load_annotation(tmp_path, "device_mask").dtype == np.uint8
    assert back.steps[0].load_annotation(tmp_path, "device_mask").sum() > 0
    assert back.steps[0].annotations["keypoints"]["tip"]["present"] is True
    assert back.outcome.metrics["tip_target"]["success"] == back.outcome.success
    assert back.outcome.metrics["wall_safety"]["perforation_risk"] is False
    assert "flow" in back.outcome.metrics and "catheter_support" in back.outcome.metrics


def test_rollout_luminal_modality(tmp_path):
    asset = procedural.straight_tube(80.0, 4.0)
    ep = rollout_episode(asset, sensor=LuminalCamera(nu=16, nv=16, n_steps=48),
                         modality="luminal", max_steps=4)
    validate(ep)
    ep.save(tmp_path)
    bundle = CaseBundle.load(tmp_path)
    assert bundle.calibration["type"] == "scope"
    assert bundle.calibration["intrinsics"]["fov_deg"] == 90.0
    rgb = bundle.episode.steps[0].load_obs(tmp_path)
    assert rgb.shape == (16, 16, 3) and rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_every_skips_render_but_keeps_kinematics():
    asset = procedural.straight_tube(80.0, 2.0)
    ep = rollout_episode(asset, sensor=FluoroSensor(res=16, nu=20, nv=20, n_samples=30),
                         max_steps=6, every=2)
    validate(ep)
    rendered = [i for i, s in enumerate(ep.steps) if s.obs_modality == "fluoro"]
    skipped = [i for i, s in enumerate(ep.steps) if s.obs_modality == "none"]
    assert rendered == [i for i in range(len(ep.steps)) if i % 2 == 0]
    assert all(ep.steps[i].obs_ref is None for i in skipped)         # none-steps carry no ref
    assert all(ep.steps[i].annotations == {} for i in skipped)       # and no fake CV labels
    assert all("device_mask_ref" in ep.steps[i].annotations for i in rendered)
    assert all("tip_mm" in s.kinematics for s in ep.steps)           # but kinematics every step


def test_meta_records_device_and_sensor_knobs():
    asset = procedural.straight_tube(80.0, 2.0)
    ep = rollout_episode(asset, sensor=FluoroSensor(res=16, nu=20, nv=20, n_samples=30),
                         max_steps=3)
    # M1: a replay/calibration harness needs the contact knobs that shape the kinematics
    for k in ("radius", "n_nodes", "node_spacing", "kappa", "d_hat", "vbd_iterations"):
        assert k in ep.meta.device
    assert ep.meta.sensor["n_samples"] == 30           # render knob recorded


def test_rollout_propagates_asset_provenance_without_embedded_external_asset():
    asset = procedural.straight_tube(80.0, 2.0)
    asset.provenance = "patient(private)"
    ep = rollout_episode(asset, sensor=FluoroSensor(res=16, nu=20, nv=20, n_samples=30),
                         max_steps=2, asset_ref="s3://private/case.asset.json")

    assert ep.meta.provenance == "patient(private)"
    assert ep.asset is None
    validate(ep)


def _build_sim(n_envs=1):
    from lumen.newton.sim import NewtonGuidewireSim
    asset = procedural.straight_tube(80.0, 2.0)
    pts, lumen = asset.edge_arrays(asset.edges[0])
    sim = NewtonGuidewireSim(np.asarray(pts), float(np.asarray(lumen.R).mean()),
                             np.asarray(pts)[:8], lumen_field=lumen, vbd_iterations=4,
                             n_envs=n_envs)
    return sim, lumen


def test_recorder_rejects_bad_modality_config():
    sim, _ = _build_sim()
    with pytest.raises(ValueError):
        EpisodeRecorder(sim, sensor=None, modality="fluoro")        # renderer required
    with pytest.raises(ValueError):
        EpisodeRecorder(sim, sensor=LuminalCamera(), modality="luminal", lumen=None)  # lumen required


def test_recorder_rejects_multi_env_sim():
    sim, _ = _build_sim(n_envs=3)                       # H1: batched sim would mix envs
    with pytest.raises(ValueError, match="single-env"):
        EpisodeRecorder(sim, sensor=FluoroSensor(res=16, nu=20, nv=20))


def test_recorder_guards_divergence(monkeypatch):
    sim, lumen = _build_sim()
    rec = EpisodeRecorder(sim, sensor=FluoroSensor(res=16, nu=20, nv=20), modality="fluoro")
    nan_nodes = np.full((8, 3), np.nan)
    monkeypatch.setattr(sim, "body_positions", lambda: nan_nodes)   # M3: force a blown-up sim
    with pytest.raises(SimDiverged):
        rec.record_step({"insertion": 1.0})
    assert rec.steps == []                              # the garbage step is NOT appended
