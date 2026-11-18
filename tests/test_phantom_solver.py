"""Tests for the CPU phantom replay adapter (lumen.experiments.phantom_solver).

The adapter replays measured phantom samples (absolute wire insertion mm +
absolute unwrapped roll deg, optionally coaxial catheter) through the pinned
procedural-bifurcation Newton tree sim on CPU, via the existing deployment
interface (``run_deployment``). Every sample must get an output entry, the
first being the documented reset state; unsupported cadence, bounds, or
deltas are REJECTED (ValueError), never clipped or saturated.
"""

import math

import numpy as np
import pytest

from lumen.experiments import phantom_solver as ps


# ---------------------------------------------------------------------------
# Validation / conversion (pure, no sim)
# ---------------------------------------------------------------------------


def test_known_initial_state_is_documented_and_required():
    assert ps.INITIAL_WIRE == [0.0, 0.0]
    assert ps.INITIAL_CATHETER == [0.0, 0.0]
    assert ps.SUPPORTED_INTERVAL_S == 0.020


def test_conversion_applies_signs_to_deltas_and_deg_to_rad():
    samples = [
        {"t_s": 0.0, "wire": [0.0, 0.0], "catheter": None},
        {"t_s": 0.020, "wire": [1.0, 30.0], "catheter": None},
        {"t_s": 0.040, "wire": [1.0, 15.0], "catheter": None},
    ]
    cmds = ps.convert_samples(samples, {"wire": [1, -1], "catheter": [1, 1]})
    assert len(cmds) == 2
    ins0, tw0 = cmds[0]
    ins1, tw1 = cmds[1]
    assert ins0 == pytest.approx(1.0)  # +1 mm insertion
    assert tw0 == pytest.approx(math.radians(-30.0))
    assert ins1 == pytest.approx(0.0)  # absolute value unchanged -> no motion
    assert tw1 == pytest.approx(math.radians(15.0))  # sign flips the roll delta


def test_rejects_unsupported_cadence():
    samples = [
        {"t_s": 0.0, "wire": [0.0, 0.0], "catheter": None},
        {"t_s": 0.010, "wire": [1.0, 0.0], "catheter": None},
    ]
    with pytest.raises(ValueError, match="cadence"):
        ps.convert_samples(samples, {"wire": [1, 1], "catheter": [1, 1]})


def test_rejects_nonuniform_cadence():
    samples = [
        {"t_s": 0.0, "wire": [0.0, 0.0], "catheter": None},
        {"t_s": 0.020, "wire": [1.0, 0.0], "catheter": None},
        {"t_s": 0.041, "wire": [2.0, 0.0], "catheter": None},
    ]
    with pytest.raises(ValueError, match="cadence"):
        ps.convert_samples(samples, {"wire": [1, 1], "catheter": [1, 1]})


def test_rejects_unknown_initial_state():
    samples = [
        {"t_s": 0.0, "wire": [5.0, 0.0], "catheter": None},
        {"t_s": 0.020, "wire": [6.0, 0.0], "catheter": None},
    ]
    with pytest.raises(ValueError, match="initial"):
        ps.convert_samples(samples, {"wire": [1, 1], "catheter": [1, 1]})


def test_rejects_position_out_of_bounds():
    samples = [
        {"t_s": 0.0, "wire": [0.0, 0.0], "catheter": None},
        {"t_s": 0.020, "wire": [ps.MAX_WIRE_MM + 1.0, 0.0], "catheter": None},
    ]
    with pytest.raises(ValueError, match="bound"):
        ps.convert_samples(samples, {"wire": [1, 1], "catheter": [1, 1]})


def test_rejects_roll_out_of_bounds():
    samples = [
        {"t_s": 0.0, "wire": [0.0, 0.0], "catheter": None},
        {"t_s": 0.020, "wire": [1.0, ps.MAX_ROLL_ABS_DEG + 1.0], "catheter": None},
    ]
    with pytest.raises(ValueError, match="bound"):
        ps.convert_samples(samples, {"wire": [1, 1], "catheter": [1, 1]})


def test_rejects_oversized_step_delta_instead_of_clipping():
    samples = [
        {"t_s": 0.0, "wire": [0.0, 0.0], "catheter": None},
        {"t_s": 0.020, "wire": [ps.MAX_STEP_INSERTION_MM + 0.1, 0.0], "catheter": None},
    ]
    with pytest.raises(ValueError, match="step"):
        ps.convert_samples(samples, {"wire": [1, 1], "catheter": [1, 1]})


def test_rejects_mixed_catheter_presence():
    samples = [
        {"t_s": 0.0, "wire": [0.0, 0.0], "catheter": None},
        {"t_s": 0.020, "wire": [1.0, 0.0], "catheter": [0.0, 0.0]},
    ]
    with pytest.raises(ValueError, match="catheter"):
        ps.convert_samples(samples, {"wire": [1, 1], "catheter": [1, 1]})


@pytest.mark.parametrize("bad", [[0, 1], [2, 1], [1.0, 1], [1, 0]])
def test_rejects_invalid_signs(bad):
    samples = [
        {"t_s": 0.0, "wire": [0.0, 0.0], "catheter": None},
        {"t_s": 0.020, "wire": [1.0, 0.0], "catheter": None},
    ]
    with pytest.raises(ValueError):
        ps.convert_samples(samples, {"wire": bad, "catheter": [1, 1]})


def test_rejects_empty_samples():
    with pytest.raises(ValueError):
        ps.convert_samples([], {"wire": [1, 1], "catheter": [1, 1]})


# ---------------------------------------------------------------------------
# Real replay through the CPU Newton sim (exercises run_deployment)
# ---------------------------------------------------------------------------


def _straight_insertion_samples(n=8, step_mm=1.0):
    return [
        {"t_s": round(i * 0.020, 6), "wire": [i * step_mm, 0.0], "catheter": None} for i in range(n)
    ]


def test_replay_returns_one_entry_per_sample_with_reset_first():
    pytest.importorskip("newton")
    out = ps.replay(_straight_insertion_samples(6), {"wire": [1, 1], "catheter": [1, 1]})
    assert len(out) == 6
    assert out[0]["event"] == "reset"
    assert all(e["event"] == "sample" for e in out[1:])
    assert out[0]["catheter"] is None
    for e in out:
        assert set(e) == {"t_s", "event", "wire", "catheter"}
        assert len(e["wire"]) == 2
        assert all(np.isfinite(e["wire"]))


def test_replay_reset_entry_matches_known_initial_state():
    pytest.importorskip("newton")
    out = ps.replay(_straight_insertion_samples(4), {"wire": [1, 1], "catheter": [1, 1]})
    x0, z0 = out[0]["wire"]
    # Pinned bifurcation: trunk runs along +z from the origin, so the reset tip
    # sits at x ~ 0 with z ~ the shaped-tip length (>= 0, small vs the route).
    assert x0 == pytest.approx(1.45, abs=0.3)
    assert 0.0 <= z0 < 25.0


def test_replay_insertion_drives_tip_downstream():
    pytest.importorskip("newton")
    out = ps.replay(
        _straight_insertion_samples(10, step_mm=1.5), {"wire": [1, 1], "catheter": [1, 1]}
    )
    zs = [e["wire"][1] for e in out]
    assert zs[-1] > zs[0] + 3.0  # real forward motion along the trunk (+z)
    assert all(z2 >= z1 - 0.5 for z1, z2 in zip(zs[:-1], zs[1:]))  # no large backslide


def test_replay_coaxial_catheter_moves_and_reports():
    pytest.importorskip("newton")
    samples = [
        {"t_s": round(i * 0.020, 6), "wire": [i * 1.0, 0.0], "catheter": [i * 0.5, 0.0]}
        for i in range(6)
    ]
    out = ps.replay(samples, {"wire": [1, 1], "catheter": [1, 1]})
    assert len(out) == 6
    assert all(e["catheter"] is not None and len(e["catheter"]) == 2 for e in out)
    cz = [e["catheter"][1] for e in out]
    assert cz[-1] > cz[0] + 0.3  # catheter insertion actually actuates


def test_replay_single_sample_is_reset_only_without_sim_steps():
    pytest.importorskip("newton")
    out = ps.replay(
        [{"t_s": 0.0, "wire": [0.0, 0.0], "catheter": None}], {"wire": [1, 1], "catheter": [1, 1]}
    )
    assert len(out) == 1 and out[0]["event"] == "reset"


def test_interface_satisfies_deployment_protocol():
    pytest.importorskip("newton")
    iface = ps.PhantomReplayInterface(coaxial=False)
    for attr in ("observe", "apply_action", "telemetry", "safe_stop"):
        assert callable(getattr(iface, attr))
    obs = iface.observe()
    assert np.isfinite(np.asarray(obs, dtype=float)).all()
    tel = iface.telemetry()
    assert "penetration_mm" in tel
    iface.safe_stop()  # must be safe to call (no-op without hardware)
