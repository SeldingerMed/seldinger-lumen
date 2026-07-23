"""Lightweight validation tests for episode capture configuration."""

import numpy as np
import pytest

from lumen.data.capture import EpisodeRecorder


class _StubSim:
    n_envs = 1
    contact_frame = object()


@pytest.mark.parametrize(
    "dt",
    [0, -0.01, float("nan"), float("inf"), 10**400, "invalid", None],
)
def test_recorder_rejects_invalid_dt(dt):
    with pytest.raises(ValueError, match="dt must be a finite positive number"):
        EpisodeRecorder(_StubSim(), modality="none", dt=dt)


@pytest.mark.parametrize("substeps", [0, -1, 1.5, True, "2", None])
def test_recorder_rejects_invalid_substeps(substeps):
    with pytest.raises(ValueError, match="substeps must be a positive integer"):
        EpisodeRecorder(_StubSim(), modality="none", substeps=substeps)


def test_recorder_normalizes_valid_numeric_timing_values():
    recorder = EpisodeRecorder(
        _StubSim(),
        modality="none",
        dt=np.float32(0.01),
        substeps=np.int64(3),
    )

    assert recorder.dt == pytest.approx(0.01)
    assert recorder.substeps == 3
    assert isinstance(recorder.dt, float)
    assert isinstance(recorder.substeps, int)
