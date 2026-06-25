"""Newton-platformed Layer 0 (doc §3.2).

The guidewire is a Newton ``add_rod`` cable integrated by ``TubeVBDSolver`` — a
fork of ``newton.solvers.SolverVBD`` that injects the tube-intrinsic contact
barrier (force + Hessian) natively into the per-color AVBD solve, so contact is
implicit and stable. Requires ``newton`` (installed from github.com/newton-physics
/newton); runs on the Warp CPU device and on CUDA.
"""

from lumen.newton.sim import NewtonGuidewireSim
from lumen.newton.tube_vbd import TubeVBDSolver
from lumen.newton.flow import NewtonFlow, FlowParams, FlowField, FlowFieldParams
from lumen.newton.hgo_wall import HGOParams
from lumen.newton.clot import ClotField, ClotParams
from lumen.newton.devices import Stentriever, FlowDiverter
from lumen.newton.aneurysm import Aneurysm, AneurysmSac

__all__ = ["NewtonGuidewireSim", "TubeVBDSolver", "NewtonFlow", "FlowParams",
           "FlowField", "FlowFieldParams",
           "HGOParams", "ClotField", "ClotParams", "Stentriever",
           "FlowDiverter", "Aneurysm", "AneurysmSac"]
