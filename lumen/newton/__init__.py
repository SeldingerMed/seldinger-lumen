"""Newton-platformed Layer 0 (doc §3.2).

The guidewire is a Newton ``add_rod`` cable integrated by ``TubeVBDSolver`` — a
fork of ``newton.solvers.SolverVBD`` that injects the tube-intrinsic contact
barrier (force + Hessian) natively into the per-color AVBD solve, so contact is
implicit and stable. Requires ``newton`` (installed from github.com/newton-physics
/newton); runs on the Warp CPU device and on CUDA.

Imports are intentionally lazy so NumPy-only helpers under ``lumen.newton`` remain
importable in development environments that have not installed Warp/Newton yet.
"""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "NewtonGuidewireSim": "lumen.newton.sim",
    "TubeVBDSolver": "lumen.newton.tube_vbd",
    "NewtonFlow": "lumen.newton.flow",
    "FlowParams": "lumen.newton.flow",
    "FlowField": "lumen.newton.flow",
    "FlowFieldParams": "lumen.newton.flow",
    "HGOParams": "lumen.newton.hgo_wall",
    "ClotField": "lumen.newton.clot",
    "ClotParams": "lumen.newton.clot",
    "Stentriever": "lumen.newton.devices",
    "FlowDiverter": "lumen.newton.devices",
    "Aneurysm": "lumen.newton.aneurysm",
    "AneurysmSac": "lumen.newton.aneurysm",
    "measure_throughput": "lumen.newton.throughput",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted([*globals(), *__all__])
