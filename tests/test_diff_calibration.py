"""Differentiable path (doc §3.5.7): Warp-autodiff gradients through the HGO
constitutive model recover planted parameters by gradient descent (calibration)."""

import numpy as np
import pytest

pytest.importorskip("warp")

from lumen.accurate.diff import calibrate_hgo, hgo_pressure_curve


def test_calibration_recovers_planted_hgo_stiffness():
    C10_true, k1_true = 8.0e3, 5.0e3
    w = np.linspace(0.05, 1.0, 16)                    # span low->high stretch (fibers engage)
    p_target = hgo_pressure_curve(w, C10_true, k1_true)
    (C10, k1), loss, hist = calibrate_hgo(w, p_target, init=(3.0e3, 1.5e3),
                                          lr=0.25, iters=800)
    assert hist[-1] < 0.05 * hist[0]                  # gradients actually reduced the loss
    assert abs(C10 - C10_true) / C10_true < 0.10      # dominant param recovered tightly
    assert abs(k1 - k1_true) / k1_true < 0.15         # fiber stiffness recovered too


def test_loss_monotonically_improves_from_a_bad_guess():
    w = np.linspace(0.05, 1.0, 12)
    p_target = hgo_pressure_curve(w, 6.0e3, 4.0e3)
    _, final, hist = calibrate_hgo(w, p_target, init=(2.0e3, 1.0e3), lr=0.25, iters=300)
    assert final < hist[0]                            # the autodiff descent works at all
    assert min(hist) <= final + 1e-6                  # final is near the best seen (converging)
