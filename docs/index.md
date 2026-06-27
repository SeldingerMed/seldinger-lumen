---
title: lumen
---

# lumen

**A differentiable, GPU-parallel solver for a continuum instrument inside a
deformable lumen.**

A guidewire in a vessel, a scope in an airway, an endoscope in a bowel — the same
physics: a slender device threading a soft, moving tube. `lumen` solves that one
problem and stays modality-agnostic. It is **Layer 0** of the
[Seldinger](https://github.com/SeldingerMed) embodied-medical-AI stack, built *on*
the [NVIDIA Newton](https://github.com/newton-physics/newton) engine.

[View on GitHub](https://github.com/SeldingerMed/seldinger-lumen){: .btn }

## Install

```bash
pip install -e ".[dev]"
```

`.[dev]` includes tests, Gymnasium, Warp, and the pinned Newton commit this solver is
validated against. For runtime-only solver use, install `.[solver]` instead. Runs on
**CPU and CUDA** from the same code (Warp picks the device at runtime).

## A 20-second taste

```python
import numpy as np
from lumen.assets import procedural
from lumen.newton.sim import NewtonGuidewireSim

asset = procedural.straight_tube(length=80, radius=2.0)
pts, lumen = asset.edge_arrays(asset.edges[0])
device = np.stack([np.full(11, 1.0), np.zeros(11), np.linspace(4, 24, 11)], axis=1)

sim = NewtonGuidewireSim(pts, R=2.0, device_points=device)
sim.step(insertion=1.0)
```

## What it models

- **Tube-intrinsic contact** injected (force + Hessian) into Newton's AVBD solve — implicit and stable.
- **HGO deformable wall** as the shared lumen field `R(s,θ)=R0+w`.
- **Anisotropic, fiber-aligned friction** and **torsion**.
- A real **clot** (Ogden, progressive damage, stent-retriever capture) and a **1-D flow pressure field**.
- CV-ready observations: contrast/vessel DRR, biplanar fluoro, masks/keypoints,
  luminal texture/artifacts, and PNG/AVI previews.
- **Accurate-tier cross-validation** against analytic ground truth.

## Learn more

- [Architecture & design invariants](https://github.com/SeldingerMed/seldinger-lumen/blob/master/ARCHITECTURE.md)
- [Contributing](https://github.com/SeldingerMed/seldinger-lumen/blob/master/CONTRIBUTING.md)

---

Apache-2.0 · clean-room · no CathSim, no patient data (enforced in CI).
