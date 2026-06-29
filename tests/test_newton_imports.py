"""Import smoke tests for optional Newton/Warp dependencies."""

from __future__ import annotations


def test_newton_package_import_is_lazy_without_backend():
    import lumen.newton as newton_pkg

    assert "HGOParams" in newton_pkg.__all__


def test_newton_numpy_helpers_import_without_backend():
    from lumen.newton.hgo_wall import HGOParams, hgo_psi

    assert hgo_psi(1.0, HGOParams()) == 0.0
