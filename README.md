# lumen

A differentiable, GPU-parallel **contact & coupling solver for a continuum
instrument inside a deformable lumen**.

The core models one abstraction: *a slender device in a deformable tube, observed
through some sensor.* That abstraction is medical-modality-agnostic — a lumen is a
blood vessel, an airway, a bowel, or a ureter depending only on which **profile**
you load. Endovascular intervention is the lead procedure, but it is one plugin
over a generic core, not the architecture.

This is the open-core substrate (Layer 0) of the Seldinger embodied-medical-AI
stack. It is built as a [Newton](https://github.com/newton-physics/newton) custom
solver — it does **not** reimplement a physics engine.

> Status: **P0** — geometric core + asset seam. The Warp contact kernels, the
> deformable shell, and the Newton solver registration land in P1–P2. See
> `ARCHITECTURE.md` and the roadmap below.

## Why it's structured this way

Three things swap to repurpose the solver across procedures, with **no core
change**:

| Swap | Where | Endovascular | Bronchoscopy | GI endoscopy |
|---|---|---|---|---|
| anatomy / lumen field `R(s,θ,t)` | `lumen.core.lumen_field` | vessel | airway | bowel |
| instrument (continuum rod) | `lumen.core.instrument` | guidewire | scope | endoscope |
| sensor (observation) | `lumen.sensors` | X-ray (projective) | RGB (luminal) | RGB (luminal) |

A new modality is a new directory under `lumen/profiles/` — see
`lumen/profiles/endovascular/`.

## Install

```bash
pip install -e ".[dev]"        # geometry core + torch fallback + tests
pip install -e ".[gpu]"        # adds warp-lang for the GPU contact kernels
```

## Layer 0 on Newton (the faithful implementation, `lumen.newton`)

The bible's Layer 0 (§3) is implemented on the **Newton** engine (the doc's build
target, §3.2 — "a domain-specialized module inside the engine"). Requires
`newton` (install from github.com/newton-physics/newton); runs on Warp-CPU and CUDA.

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
  analytic ground truth to ~1e-6; STARK/ppf-contact-solver drop-in slot.
- **GPU-validated** on an RTX 3090 (contact holds, HGO wall deflects, cross-val to
  3e-6, 31 steps/s full stack).

```python
from lumen.newton.sim import NewtonGuidewireSim   # needs newton + warp
```

## Backends (Warp/Newton primary, PyTorch fallback)

The contact narrowphase + analytic barrier are implemented as **Warp kernels**
(`lumen.physics.warp_contact`), differentiable via Warp's autodiff tape, batched
across environments. `lumen.physics.backend.select_backend()` is hardware-gated:

```
warp-cuda  →  warp-cpu  →  torch     (favour Warp/Newton; torch is the fallback)
```

The same kernel source runs on CUDA and on the Warp CPU device; the PyTorch
solver (`lumen.physics.contact`) is the always-available fallback and the
reference the Warp path is kept in parity with (`tests/test_warp_parity.py`).

GPU-validated on an RTX 3090 (CUDA path, `benchmarks/warp_gpu_check.py`):
CPU↔CUDA kernel parity to 1e-7; **349M node-evals/s ≈ 21.8M env-steps/s** at
B=65536 (~140× over the Warp CPU device; the doc's ≥1e4 env-steps/s target, §3.1,
beaten by 3 orders of magnitude). Full Warp rod + solver integration (today the
stepper is torch; Warp supplies the GPU narrowphase) is the next step, with
Newton VBD as the rod substrate.

## Quickstart

```python
import numpy as np
from lumen.assets import procedural
from lumen.core.frame import CenterlineFrame

asset = procedural.straight_tube(length=100, radius=2.0)
pts, lumen = asset.edge_arrays(asset.edges[0])
frame = CenterlineFrame(pts)

p = frame.project(np.array([1.0, 0.0, 50.0]))   # world point -> tube-intrinsic
print(p.s, p.theta, p.r)                         # arc-length, angle, radius
print(lumen.gap(p.s, p.theta, p.r))              # contact gap R - r (>0 = clearance)
```

## Layout

```
lumen/core/       frame · lumen_field   (+ instrument · wall · contact · coupling · solver, P1+)
lumen/sensors/    projective | luminal | wave observation models
lumen/profiles/   endovascular | bronchoscopy | …   (the repurposing surface)
lumen/assets/     schema (the integration seam) · procedural generator
tools/            firewall check (no CathSim, no patient data)
```

## Roadmap

- **P0 ✅** repo + asset seam + tube-intrinsic projection
- **P1 ✅** contact narrowphase + analytic barrier; rod in a tube; friction
  recovered via gradient (`lumen.physics`, `python -m lumen.physics.sysid`)
- **P2 ✅** deformable anisotropic shell sharing `R`; coupled rod–soft contact
  (point-load deflection matches analytic Winkler; HGO-style axial/hoop anisotropy)
- **P3 ✅** differentiable physics → projective sensor; device-as-sensor recovers
  a mechanical parameter from the fluoro image alone (`lumen.physics.imaging`)
- **P4 ✅** one-way Windkessel flow drag + generic clot/occlusion interface
  (R-collapse + adhesive capture; real INSIST/Luraghi model plugs in privately)
- **P5 ✅** Gym nav env (`lumen.envs.NavEnv`); benchmark suite + JSON leaderboard;
  narrowphase throughput benchmark; rod-primitive bake-off harness (`benchmarks/`)

```bash
python -m lumen.physics.sysid       # M0: recover friction by gradient
python -m lumen.physics.imaging     # M2: device-as-sensor (param from fluoro image)
python -m benchmarks.throughput     # narrowphase: flat in wall resolution T
python -m benchmarks.leaderboard    # navigation leaderboard (JSON)
```

> The physics tier is PyTorch (differentiable, batched, CPU/MPS/CUDA). A Warp
> kernel port is the documented GPU-throughput upgrade; the *formulation* is
> substrate-independent. Backprop-through-time through stiff contact corrupts past
> ~50 steps (doc §3.5.7) — calibration uses short differentiable horizons.

## License & boundaries

Apache-2.0. This repo is deliberately **clean-room**: it depends only on
Newton/Warp, never on CathSim (CC-BY-NC-SA-4.0), and contains **no patient data** —
every asset is procedurally generated. Patient pipelines and real-data
calibration live in the private Seldinger repos behind the `lumen.assets.schema`
seam. CI enforces both boundaries (`tools/check_firewall.py`).
