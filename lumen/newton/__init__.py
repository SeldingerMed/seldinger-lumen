"""Newton-platformed Layer 0 (doc §3.2).

The guidewire is a Newton ``add_rod`` cable integrated by ``TubeVBDSolver`` — a
fork of ``newton.solvers.SolverVBD`` that injects the tube-intrinsic contact
barrier (force + Hessian) natively into the per-color AVBD solve, so contact is
implicit and stable. Requires ``newton`` (installed from github.com/newton-physics
/newton); runs on the Warp CPU device and on CUDA.

Some helper modules in this package are NumPy-only. Keep optional Warp/Newton
imports lazy so ``import lumen.newton.clot`` and other dependency-light imports
continue to work in environments that have not installed the solver extras.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "NewtonGuidewireSim": ("lumen.newton.sim", "NewtonGuidewireSim"),
    "TubeVBDSolver": ("lumen.newton.tube_vbd", "TubeVBDSolver"),
    "NewtonFlow": ("lumen.newton.flow", "NewtonFlow"),
    "FlowParams": ("lumen.newton.flow", "FlowParams"),
    "FlowField": ("lumen.newton.flow", "FlowField"),
    "FlowFieldParams": ("lumen.newton.flow", "FlowFieldParams"),
    "HGOParams": ("lumen.newton.hgo_wall", "HGOParams"),
    "ClotField": ("lumen.newton.clot", "ClotField"),
    "ClotParams": ("lumen.newton.clot", "ClotParams"),
    "Stentriever": ("lumen.newton.devices", "Stentriever"),
    "FlowDiverter": ("lumen.newton.devices", "FlowDiverter"),
    "Aneurysm": ("lumen.newton.aneurysm", "Aneurysm"),
    "AneurysmSac": ("lumen.newton.aneurysm", "AneurysmSac"),
    "measure_throughput": ("lumen.newton.throughput", "measure_throughput"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily resolve package-level exports.

    This preserves the public ``from lumen.newton import ...`` convenience API
    without importing Warp/Newton-backed modules during package initialization.
    """
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
