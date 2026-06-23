"""L2.1 — synthetic capture: drive a sim, log paired observations, save an episode.

Needs Newton+Warp (it runs the Layer-0 sim) → importorskip, like the other RL tests."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.assets import procedural                      # noqa: E402
from lumen.data import Episode, EpisodeRecorder, rollout_episode, validate  # noqa: E402
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

    ep.save(tmp_path)
    back = Episode.load(tmp_path)
    validate(back, root=tmp_path)                        # all sidecars present
    assert back.steps[0].load_obs(tmp_path).shape == (24, 24)
    assert back.steps[0].load_nodes(tmp_path).shape[1] == 3


def test_rollout_luminal_modality(tmp_path):
    asset = procedural.straight_tube(80.0, 4.0)
    ep = rollout_episode(asset, sensor=LuminalCamera(nu=16, nv=16, n_steps=48),
                         modality="luminal", max_steps=4)
    validate(ep)
    ep.save(tmp_path)
    rgb = Episode.load(tmp_path).steps[0].load_obs(tmp_path)
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
    assert all("tip_mm" in s.kinematics for s in ep.steps)           # but kinematics every step


def test_recorder_rejects_bad_modality_config():
    asset = procedural.straight_tube(80.0, 2.0)
    pts, lumen = asset.edge_arrays(asset.edges[0])
    from lumen.newton.sim import NewtonGuidewireSim
    sim = NewtonGuidewireSim(np.asarray(pts), float(np.asarray(lumen.R).mean()),
                             np.asarray(pts)[:8], lumen_field=lumen, vbd_iterations=4)
    with pytest.raises(ValueError):
        EpisodeRecorder(sim, sensor=None, modality="fluoro")        # renderer required
    with pytest.raises(ValueError):
        EpisodeRecorder(sim, sensor=LuminalCamera(), modality="luminal", lumen=None)  # lumen required
