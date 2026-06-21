"""Differentiable, batched physics tier (PyTorch).

This is the runnable Layer-0 solver: the genuine algorithms from the doc
(tube-intrinsic contact + analytic barrier, a deformable shell sharing R, an
imaging coupling), implemented in PyTorch so they are exactly differentiable
(autograd), batched across environments (the leading B dimension), and run on
CPU / MPS / CUDA without requiring a CUDA-only engine.

The doc's target substrate is Newton/Warp for ultimate GPU throughput; porting
these kernels to Warp is the documented throughput upgrade (see ARCHITECTURE.md).
The *formulation* -- tube-intrinsic narrowphase, shared R field, physics<->sensor
coupling -- is substrate-independent and lives here, tested.

torch is an optional extra (`pip install -e ".[physics]"`); the numpy geometry
core does not depend on it.
"""

from lumen.physics.rod import Rod, RodParams
from lumen.physics.contact import ContactGeometry, ContactParams
from lumen.physics.solver import SimConfig, Solver
from lumen.physics.wall import WallShell, WallShellParams
from lumen.physics.flow import WindkesselFlow
from lumen.physics.occlusion import Occlusion
from lumen.physics import backend

__all__ = ["Rod", "RodParams", "ContactGeometry", "ContactParams",
           "SimConfig", "Solver", "WallShell", "WallShellParams",
           "WindkesselFlow", "Occlusion", "backend"]
