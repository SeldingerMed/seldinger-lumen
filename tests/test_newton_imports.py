"""Import behavior for optional Newton/Warp solver dependencies."""

from __future__ import annotations


def test_newton_package_import_is_lazy_without_backend():
    import lumen.newton as newton_pkg

    assert "HGOParams" in newton_pkg.__all__
    assert newton_pkg.HGOParams().__class__.__name__ == "HGOParams"


def test_newton_numpy_helpers_import_without_backend():
    from lumen.newton.hgo_wall import HGOParams, hgo_psi

    assert hgo_psi(1.0, HGOParams()) == 0.0


def test_numpy_only_newton_submodules_import_without_solver_extras():
    from lumen.newton.clot import ClotField, ClotParams
    from lumen.newton.flow import FlowField, FlowFieldParams
    from lumen.newton.hgo_wall import HGOParams

    assert ClotField is not None
    assert ClotParams is not None
    assert FlowField is not None
    assert FlowFieldParams is not None
    assert HGOParams is not None


def test_package_exports_are_lazy_for_numpy_only_symbols():
    from lumen.newton import ClotParams, FlowParams, HGOParams

    assert ClotParams is not None
    assert FlowParams is not None
    assert HGOParams is not None
