"""L2.3 — sim2sim calibration from a wall-probe episode.

The math is numpy, but the device-as-sensor import chain pulls in warp/newton, so
guard the import (matches the other layer tests) — it still runs fast (no sim)."""

import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")

from lumen.data import (Episode, EpisodeDataset, calibrate_from_episode,  # noqa: E402
                        probe_episode, validate)
from lumen.sensors import FluoroSensor                                    # noqa: E402
from lumen.sensors.device_as_sensor import device_on_wall                 # noqa: E402


def _biplanar(sensor, C10=6e3):
    nodes = device_on_wall(C10)
    return [sensor.default_carm(nodes, axis=(1, 0, 0)),
            sensor.default_carm(nodes, axis=(0, 1, 0))]


def test_probe_episode_is_a_valid_calibration_episode():
    sensor = FluoroSensor(res=24, n_samples=60, nu=32, nv=32)
    ep = probe_episode(5e3, sensor, carms=_biplanar(sensor))
    validate(ep)                                            # schema-valid
    assert ep.outcome.steps == 2 and all(s.obs_modality == "fluoro" for s in ep.steps)
    assert ep.meta.notes["calib"]["true_C10"] == 5e3       # ground truth stored for sim2sim
    assert len(ep.meta.notes["calib"]["carms"]) == 2       # views self-contained (in calib, not sensor)
    assert ep.meta.notes["episode_kind"] == "wall_probe"   # kind discriminator
    assert "carms" not in ep.meta.sensor                   # meta.sensor keeps the documented shape


def test_calibration_recovers_planted_stiffness_in_memory():
    sensor = FluoroSensor(res=36, n_samples=90, nu=44, nv=44)
    ep = probe_episode(6e3, sensor, carms=_biplanar(sensor))
    res = calibrate_from_episode(ep, init_C10=2e3, iters=16)
    assert res["true_C10"] == 6e3 and res["n_views"] == 2
    assert res["rel_error"] < 0.1                          # biplanar recovers (the L1.2 regime)


def test_calibration_round_trips_through_disk(tmp_path):
    sensor = FluoroSensor(res=36, n_samples=90, nu=44, nv=44)
    probe_episode(6e3, sensor, carms=_biplanar(sensor)).save(tmp_path)
    loaded = EpisodeDataset(tmp_path)[0]                    # carms + sensor reconstructed from manifest
    res = calibrate_from_episode(loaded, init_C10=2e3, iters=16)
    assert res["rel_error"] < 0.1


def test_calibrate_rejects_navigation_episode():
    # a non-probe episode has no notes['calib'] -> clear refusal, not a garbage number
    from lumen.data import EpisodeMeta, Outcome, Step
    nav = Episode(meta=EpisodeMeta(notes={"target_s": 50.0}),
                  steps=[Step(t=0.0, obs_modality="none")], outcome=Outcome(steps=1))
    with pytest.raises(ValueError, match="calib"):
        calibrate_from_episode(nav)


def test_noise_probe_exposes_mono_underdetermination():
    # H1: noise-free recovery is trivial (loss(true)=0 exactly) for BOTH mono and
    # biplanar — so the harness must probe identifiability under noise. A mono
    # out-of-plane view blows up; biplanar holds (the §3.6 gate, THROUGH the harness).
    sensor = FluoroSensor(res=36, n_samples=90, nu=44, nv=44)
    cx, cy = _biplanar(sensor)
    mono = calibrate_from_episode(probe_episode(6e3, sensor, carms=[cx]),
                                  init_C10=2e3, iters=16, noise_std=1e-3)
    bi = calibrate_from_episode(probe_episode(6e3, sensor, carms=[cx, cy]),
                                init_C10=2e3, iters=16, noise_std=1e-3)
    assert mono["rel_error"] < 0.05 and bi["rel_error"] < 0.05          # both trivial noise-free
    assert mono["rel_error_noisy"] > 5 * bi["rel_error_noisy"]          # but mono degrades under noise
    assert bi["identifiable"] and not mono["identifiable"]              # the honest flag


def test_depth_aligned_default_view_warns():
    sensor = FluoroSensor(res=24, n_samples=60, nu=32, nv=32)
    with pytest.warns(UserWarning, match="depth-ambiguous"):
        probe_episode(6e3, sensor, view_axis=(1, 0, 0), bulge_dir=(1, 0, 0))   # view ∥ bulge


def test_calibrate_rejects_episode_missing_carms():
    sensor = FluoroSensor(res=24, n_samples=60, nu=32, nv=32)
    ep = probe_episode(6e3, sensor, carms=_biplanar(sensor))
    ep.meta.notes["calib"].pop("carms")                                 # malformed calib block
    with pytest.raises(ValueError, match="C-arm views"):
        calibrate_from_episode(ep)
