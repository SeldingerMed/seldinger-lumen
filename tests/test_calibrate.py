"""L2.3 — sim2sim calibration from a wall-probe episode.

The math is numpy, but the device-as-sensor import chain pulls in warp/newton, so
guard the import (matches the other layer tests) — it still runs fast (no sim)."""

import numpy as np
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
    assert len(ep.meta.sensor["carms"]) == 2               # views are self-contained


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


def test_mono_view_calibration_runs():
    # the harness handles a single-view probe (n_views=1). The mono-vs-biplanar
    # identifiability GAP only shows under noise and is proven at L1.2
    # (test_sensors_inverse.py); not duplicated here with a flaky noise-free inequality.
    sensor = FluoroSensor(res=36, n_samples=90, nu=44, nv=44)
    cx = _biplanar(sensor)[0]
    res = calibrate_from_episode(probe_episode(6e3, sensor, carms=[cx]), init_C10=2e3, iters=16)
    assert res["n_views"] == 1 and np.isfinite(res["rel_error"])
