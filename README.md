# lumen

**A differentiable, GPU-parallel solver for a continuum instrument inside a deformable lumen.**

A guidewire in a blood vessel, a scope in an airway, an endoscope in a bowel — all
the same physics problem: *a slender device threading a soft, moving tube.* `lumen`
solves that one problem well, and stays modality-agnostic so the same core serves
any of them. Endovascular intervention is the lead use case, not the architecture.

This is **Layer 0** of the [Seldinger](https://github.com/SeldingerMed) embodied-medical-AI
stack — the contact-and-coupling substrate everything else builds on. It runs *on*
the [NVIDIA Newton](https://github.com/newton-physics/newton) engine; it does not
reimplement one.

> **Status:** Layer 0 complete and GPU-validated — tube-intrinsic contact, HGO
> deformable wall, anisotropic friction, torsion, a real clot + 1-D flow field, and
> accurate-tier cross-validation. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## Install

```bash
pip install -e ".[dev]"    # core + pinned Newton/Warp backend + tests
```

The geometry core needs only NumPy. The solver needs Warp + Newton, which run on
**both CPU (Warp's LLVM backend) and CUDA** — same code, device picked at runtime.
There is no separate CPU fallback to maintain.

```bash
python -m lumen.hardware     # -> {"device": "cuda"|"cpu", ...}
```

## Quick start

```python
import numpy as np
from lumen.assets import procedural
from lumen.newton.sim import NewtonGuidewireSim

asset = procedural.straight_tube(length=80, radius=2.0)
pts, lumen = asset.edge_arrays(asset.edges[0])
device = np.stack([np.full(11, 1.0), np.zeros(11), np.linspace(4, 24, 11)], axis=1)

sim = NewtonGuidewireSim(pts, R=2.0, device_points=device)   # device auto-detected
sim.step(insertion=1.0)
print(sim.node_radii().max())
```

The dependency-light geometry core works standalone:

```python
from lumen.core.frame import CenterlineFrame
hit = CenterlineFrame(pts).project(np.array([1.0, 0.0, 50.0]))   # -> (s, θ, r)
```

## What's inside (`lumen.newton`)

| Piece | What it does |
|---|---|
| **Guidewire** | Newton `add_rod` cable — stretch + bend/twist; torsion carries proximal rotation to the tip |
| **`TubeVBDSolver`** | a fork of Newton's `SolverVBD` that injects the tube-intrinsic contact **barrier + Hessian** into the AVBD solve, so contact is implicit and stable |
| **HGO wall** | Holzapfel–Gasser–Ogden anisotropic shell as the deformable **shared lumen field** `R(s,θ)=R0+w` |
| **Anisotropic friction** | Coulomb friction with μ varying by slide angle to the collagen fiber direction |
| **Clot** | finite-extent Ogden occlusion that collapses the shared `R`; progressive damage → fragmentation; stent-retriever capture/slip/fragment |
| **Flow** | 1-D resistive pressure field `P(s)`/`v(s)` along the centerline (clot raises resistance, aspiration is a pressure sink), with a lumped Windkessel fallback |
| **Cross-validation** | fast-tier kernels vs. analytic ground truth to ~1e-6; STARK / ppf-contact-solver drop-in slot |

## Layout

```
lumen/core/       frame · lumen_field        tube-intrinsic geometry (NumPy only)
lumen/newton/     sim · tube_vbd · tube_barrier_kernel · hgo_wall · clot · flow · devices · crossval
lumen/assets/     schema (the private-data seam) · procedural generator
lumen/profiles/   endovascular | …           the repurposing surface
lumen/envs/       NavEnv (Gym, Newton-backed)
lumen/hardware.py device detection (cuda/cpu)
tools/            firewall check (no CathSim, no patient data)
```

## License & boundaries

[Apache-2.0](LICENSE), deliberately **clean-room**:

- **No [CathSim](https://github.com/robotvisionlabs/cathsim)** (CC-BY-NC-SA-4.0 would
  contaminate the license).
- **No patient data** — every committed asset is procedurally generated.

Both are enforced in CI by `tools/check_firewall.py`. Patient pipelines and
real-data calibration live in the private Seldinger repos behind the
`lumen.assets.schema` seam, layered *on top of* this open core.

## Contributing

We welcome contributions — see [CONTRIBUTING.md](CONTRIBUTING.md). In short: sign
your commits off (`git commit -s`, [DCO](https://developercertificate.org/)), keep
`pytest` and the firewall green, and open a PR. New modalities are new directories
under `lumen/profiles/` — the core never changes.
