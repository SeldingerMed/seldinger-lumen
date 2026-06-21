# lumen

A differentiable, GPU-parallel **contact & coupling solver for a continuum
instrument inside a deformable lumen**.

The core models one abstraction: *a slender device in a deformable tube.* That
abstraction is medical-modality-agnostic — a lumen is a blood vessel, an airway, a
bowel, or a ureter depending only on which **profile** you load. Endovascular
intervention is the lead procedure, but it is one plugin over a generic core, not
the architecture.

This is the open-core substrate (Layer 0) of the Seldinger embodied-medical-AI
stack. It is built **on the [Newton](https://github.com/newton-physics/newton)
engine** (the doc's build target, §3.2: "a domain-specialized module inside the
engine") — it does **not** reimplement a physics engine.

> Status: **Layer 0 complete on Newton** — guidewire + tube-intrinsic contact,
> HGO deformable wall, anisotropic friction, torsion, clot + two-way flow,
> accurate-tier cross-validation; GPU-validated. See `ARCHITECTURE.md`.

## One engine, CPU and GPU (no separate fallback)

Newton runs on **both** the CPU (Warp's LLVM backend) and **CUDA** GPUs — the same
solver code, picked at runtime:

```bash
python -m lumen.hardware     # -> {"device": "cuda"|"cpu", "warp": ..., "newton": ...}
```

`NewtonGuidewireSim` calls `lumen.hardware.detect_device()` for its default
(`cuda` if a GPU is visible to Warp, else `cpu`). There is no PyTorch fallback —
it was removed because Newton already covers CPU; a parallel engine would violate
"do not write an engine" (§3.2).

## Why it's structured this way

Three things swap to repurpose the solver across procedures, with **no core
change** (doc §3.9):

| Swap | Where | Endovascular | Bronchoscopy | GI endoscopy |
|---|---|---|---|---|
| anatomy / lumen field `R(s,θ,t)` | `lumen.core.lumen_field`, `lumen.newton.hgo_wall` | vessel | airway | bowel |
| instrument (continuum rod) | `lumen.newton` (Newton cable) | guidewire | scope | endoscope |
| sensor (observation, Layer 1) | *future* (projective X-ray / luminal RGB) | X-ray | RGB | RGB |

A new modality is a new directory under `lumen/profiles/`.

## Install

```bash
pip install -e ".[dev]"                                   # core + warp + tests
pip install "git+https://github.com/newton-physics/newton"   # the Newton engine
```

## Layer 0 on Newton (`lumen.newton`)

- **Guidewire** = Newton `add_rod` cable (stretch + bend/twist); **torsion**
  transmits proximal rotation to the distal tip (whip).
- **`TubeVBDSolver`** (`vbd_fork.py`) — a fork of Newton's `SolverVBD` that injects
  the **tube-intrinsic contact barrier (force + Hessian)** into the per-color AVBD
  solve, so contact is implicit and stable (not an external force).
- **HGO wall** (`hgo_wall.py`) — Holzapfel-Gasser-Ogden anisotropic hyperelastic
  shell as the **deformable shared lumen field** R(s,θ)=R0+w; contact reads R_eff
  and deposits load; the wall deforms per HGO.
- **Anisotropic friction** — Coulomb friction with μ varying by the slide angle to
  the collagen fiber direction.
- **Clot** (`clot.py`) — INSIST/Luraghi Ogden clot, adhesive/frictional capture,
  fragmentation criterion, **two-way aspiration/flow** coupling.
- **Accurate-tier cross-validation** (`crossval.py`) — fast-tier kernels vs
  analytic ground truth to ~1e-6; STARK/ppf-contact-solver drop-in slot (§3.3).
- **GPU-validated** on an RTX 3090 (contact holds, HGO wall deflects, cross-val to
  3e-6).

```python
import numpy as np
from lumen.assets import procedural
from lumen.newton.sim import NewtonGuidewireSim     # needs newton + warp

asset = procedural.straight_tube(length=80, radius=2.0)
pts, lumen = asset.edge_arrays(asset.edges[0])
dev = np.stack([np.full(11, 1.0), np.zeros(11), np.linspace(4, 24, 11)], axis=1)
sim = NewtonGuidewireSim(pts, 2.0, dev)              # device auto-detected
sim.step(insertion=1.0)
print(sim.node_radii().max())
```

The geometry core is dependency-light (numpy only):

```python
from lumen.core.frame import CenterlineFrame
p = CenterlineFrame(pts).project(np.array([1.0, 0.0, 50.0]))   # -> (s, θ, r)
```

## Layout

```
lumen/core/       frame · lumen_field        (tube-intrinsic geometry, numpy)
lumen/newton/     sim · vbd_fork · tube_barrier_kernel · hgo_wall · clot · crossval
lumen/assets/     schema (the integration seam) · procedural generator
lumen/profiles/   endovascular | …           (the repurposing surface)
lumen/envs/       NavEnv (Gym, Newton-backed)
lumen/hardware.py device detection (cuda/cpu)
tools/            firewall check (no CathSim, no patient data)
```

## Milestones (doc §3.10), as implemented on Newton

- **M0** guidewire + tube-intrinsic contact in the AVBD solve (implicit, stable)
- **M1** HGO deformable wall sharing R; rod–soft contact
- **M2** accurate-tier cross-validation (analytic oracle; STARK/ppf drop-in)
- **M3** clot constitutive (INSIST/Luraghi) + two-way flow / aspiration
- **M4** anisotropic fiber-aligned friction; torsion/whip
- **M5** Gym `NavEnv` + JSON leaderboard (`benchmarks/leaderboard.py`)

## License & boundaries

Apache-2.0. Deliberately **clean-room**: depends only on Newton/Warp, never on
CathSim (CC-BY-NC-SA-4.0), and contains **no patient data** — every asset is
procedurally generated. Patient pipelines and real-data calibration (HGO/clot
parameters) live in the private Seldinger repos behind the `lumen.assets.schema`
seam. CI enforces both boundaries (`tools/check_firewall.py`).
