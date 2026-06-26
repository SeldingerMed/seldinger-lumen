"""L1.2 / M2 — the friction arm of the device-as-sensor loop, and the joint (C10, mu)
inverse with bounded-identifiability reporting. Pure numpy (renderer + HGO closed form).

Completes the M2 "calibrate wall/friction from biplanar DSA, parameters recovered with
reported, bounded identifiability" pair: wall stiffness was already invertible; this adds
the axial stick-slip lag (friction) and shows the two are separably identifiable biplanar.
"""

import numpy as np
import pytest

from lumen.sensors import FluoroSensor
from lumen.sensors.carm import CArm
from lumen.sensors.device_as_sensor import (device_with_friction, device_wall_and_friction,
                                            estimate_friction, estimate_wall_and_friction,
                                            friction_identifiability, friction_sensitivity,
                                            joint_identifiability)


def _sensor():
    return FluoroSensor(mu_device=1.0, res=32, n_samples=64)   # light enough for CI; recovery holds


def _views():
    # two complementary C-arms: +x foreshortens the lateral bulge, +y sees both in-plane
    return (CArm.looking_at([0, 0, 0], distance=120, axis=(1, 0, 0)),
            CArm.looking_at([0, 0, 0], distance=120, axis=(0, 1, 0)))


# ---- forward sanity -----------------------------------------------------------
def test_friction_forward_lags_the_tip_monotonically():
    # higher friction -> the pushed device's tip advances LESS (stick-slip lag)
    tips = [device_with_friction(mu)[-1, 2] for mu in (0.0, 0.3, 0.6, 1.0)]
    assert all(np.diff(tips) < 0)                     # strictly decreasing tip-z with mu
    assert device_with_friction(0.0)[-1, 2] > device_with_friction(1.0)[-1, 2] + 1.0  # visible


def test_friction_forward_rejects_negative_mu():
    with pytest.raises(ValueError, match="mu must be"):
        device_with_friction(-0.1)


def test_forward_and_optimizer_input_guards():
    # the review's fail-fast guards: bad geometry / confounded forward / invalid seeds
    with pytest.raises(ValueError, match="must be > 0"):
        device_with_friction(0.5, k_axial=0.0)
    with pytest.raises(ValueError, match="~parallel"):        # bulge ∥ lag -> view-independent confound
        device_wall_and_friction(3e3, 0.5, bulge_dir=(0, 0, 1), axis=(0, 0, 1))
    s, (cx, _) = _sensor(), _views()
    with pytest.raises(ValueError, match="init_mu"):
        estimate_friction([np.zeros((8, 8))], s, [cx], init_mu=-0.1)
    with pytest.raises(ValueError, match="init_C10"):
        estimate_wall_and_friction([np.zeros((8, 8))], s, [cx], init_C10=-1.0)


def test_friction_sensitivity_nonzero_at_mu_zero_boundary():
    # a rel·mu step would be 0 at mu=0; the rel·max(mu,0.1) step keeps it informative
    s, (_, cy) = _sensor(), _views()
    assert friction_sensitivity(0.0, s, [cy]) > 0.0


# ---- friction recovery --------------------------------------------------------
def test_estimate_friction_recovers_mu():
    s, (cx, cy) = _sensor(), _views()
    true_mu = 0.5
    targets = [s.render(device_with_friction(true_mu), carm=c)[0] for c in (cx, cy)]
    mu_hat, hist = estimate_friction(targets, s, [cx, cy], init_mu=0.2, iters=20)
    assert hist[-1] < 0.05 * hist[0]                  # image loss collapsed
    assert abs(mu_hat - true_mu) < 0.05               # mu recovered


def test_friction_sensitivity_grows_with_views():
    s, (cx, cy) = _sensor(), _views()
    mono = friction_sensitivity(0.5, s, [cy])
    bi = friction_sensitivity(0.5, s, [cx, cy])
    assert bi > mono > 0                              # more views -> more identifiable


def test_friction_identifiability_curves_have_a_minimum_at_truth():
    s, (cx, cy) = _sensor(), _views()
    grid = np.linspace(0.1, 1.0, 10)
    out = friction_identifiability(0.5, s, {"biplanar": [cx, cy]}, grid)
    losses = out["biplanar"]
    assert grid[int(np.argmin(losses))] == pytest.approx(0.5, abs=0.12)  # min near truth


# ---- joint (C10, mu) ----------------------------------------------------------
def test_estimate_wall_and_friction_recovers_both_biplanar():
    s, (cx, cy) = _sensor(), _views()
    true_C10, true_mu = 3.0e3, 0.5
    targets = [s.render(device_wall_and_friction(true_C10, true_mu), carm=c)[0] for c in (cx, cy)]
    C10_hat, mu_hat, hist = estimate_wall_and_friction(targets, s, [cx, cy],
                                                       init_C10=5e3, init_mu=0.2, iters=30)
    assert hist[-1] < 0.05 * hist[0]
    assert abs(C10_hat - true_C10) / true_C10 < 0.1   # wall stiffness recovered
    assert abs(mu_hat - true_mu) < 0.08               # friction recovered jointly


def test_joint_identifiability_is_resolved_by_biplanar():
    # the M2 gate: a single view aligned with the bulge axis is catastrophically ill-
    # conditioned (the two params confound); biplanar rescues it and — the monotone claim —
    # strictly improves the worst-determined direction (lam_min).
    s, (cx, cy) = _sensor(), _views()
    out = joint_identifiability(3.0e3, 0.5, s, {"mono_x": [cx], "biplanar": [cx, cy]})
    assert out["mono_x"]["cond"] > 50.0               # bulge foreshortened -> ill-posed
    assert out["biplanar"]["cond"] < 0.2 * out["mono_x"]["cond"]   # biplanar rescues it
    assert out["biplanar"]["lam_min"] > out["mono_x"]["lam_min"]   # monotone: worst dir improves
    assert abs(out["biplanar"]["corr"]) < 0.6         # bulge vs lag are near-orthogonal


# ---- the Layer-2 episode seam (wall+friction probe -> joint recovery) ---------
def test_joint_probe_episode_recovers_both_through_the_seam():
    from lumen.data.calibrate import calibrate_from_episode, joint_probe_episode
    s = _sensor()
    nodes = device_wall_and_friction(6.0e3, 0.6)
    cx = s.default_carm(nodes, axis=(1, 0, 0))
    cy = s.default_carm(nodes, axis=(0, 1, 0))
    ep = joint_probe_episode(6.0e3, 0.6, s, carms=[cx, cy])
    assert ep.meta.notes["episode_kind"] == "wall_friction_probe"
    res = calibrate_from_episode(ep, init_C10=3e3, init_mu=0.3, iters=30)
    assert res["rel_error_C10"] < 0.1 and res["rel_error_mu"] < 0.1    # both recovered (relative)
    assert res["n_views"] == 2
    # the deterministic Fisher gate is wired in alongside the recovery (cond/lam_min)
    assert res["fisher_cond"] > 0 and res["fisher_lam_min"] > 0
