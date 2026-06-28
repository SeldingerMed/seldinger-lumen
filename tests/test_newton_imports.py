"""Import behavior for optional Newton/Warp solver dependencies."""


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
