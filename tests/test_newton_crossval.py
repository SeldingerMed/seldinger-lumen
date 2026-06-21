"""Task #5: accurate-tier cross-validation of the fast tier (doc §3.3, §3.8).

Analytic oracle (always available) — the fast-tier Warp kernels are cross-checked
against closed-form references. STARK/ppf-contact-solver is the heavy-scene oracle
that drops in on a GPU box.
"""

import pytest

from lumen.newton.crossval import (accurate_tier_status, crossval_contact_force,
                                   crossval_hgo_stress)


def test_hgo_stress_matches_analytic():
    assert crossval_hgo_stress() < 1e-6


def test_contact_force_matches_analytic_compliant_and_log():
    pytest.importorskip("warp")
    assert crossval_contact_force(mode="compliant") < 1e-3
    assert crossval_contact_force(mode="log") < 1e-3


def test_accurate_tier_reports_oracle():
    s = accurate_tier_status()
    assert s["analytic_oracle"] is True
    assert "external_oracle" in s        # STARK/ppf drop-in slot
