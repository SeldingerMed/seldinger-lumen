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
pip install -e ".[dev]"      # geometry core + tests, no GPU needed
pip install -e ".[gpu]"      # adds warp-lang for the contact kernels (P1+)
```

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
- **P1** Warp contact narrowphase + analytic barrier; rod in rigid tube; friction
  via gradient (first tagged release `v0.1`)
- **P2** deformable anisotropic shell sharing `R`; rod–soft contact
- **P3** differentiable physics → sensor wiring (synthetic DRR)
- **P4** flow coupling + generic occlusion interface
- **P5** rod-primitive bake-off; benchmark suite + leaderboard; Isaac Lab/Gym

## License & boundaries

Apache-2.0. This repo is deliberately **clean-room**: it depends only on
Newton/Warp, never on CathSim (CC-BY-NC-SA-4.0), and contains **no patient data** —
every asset is procedurally generated. Patient pipelines and real-data
calibration live in the private Seldinger repos behind the `lumen.assets.schema`
seam. CI enforces both boundaries (`tools/check_firewall.py`).
