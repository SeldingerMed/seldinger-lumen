<p align="center">
  <img src="docs/assets/logo.svg" alt="lumen" width="760">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2dd4bf" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-5eead4" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/runs%20on-CPU%20%2B%20CUDA-7c8da0" alt="CPU + CUDA">
  <img src="https://img.shields.io/badge/engine-NVIDIA%20Newton-7c8da0" alt="NVIDIA Newton">
</p>

A guidewire in a blood vessel, a scope in an airway, an endoscope in a bowel — it's
one physics problem: **a slender device threading a soft, moving tube.** `lumen`
solves that problem on GPU, with differentiable contact, and stays modality-agnostic
so the same core handles any of them. Endovascular navigation is the lead use case;
swapping anatomy, instrument, and sensor is a config change, not a rewrite.

It runs *on* the [NVIDIA Newton](https://github.com/newton-physics/newton) engine
rather than reimplementing one. The wall, the contact, and the collision geometry are
all the same object — a tube-intrinsic radius field `R(s, θ, t)` — so the deformable
wall and the contact barrier can never drift out of sync.

## Install

```bash
pip install -e ".[dev]"     # core + pinned Newton/Warp backend + tests
```

The geometry core needs only NumPy. The solver adds Warp + Newton, which run on **CPU
(Warp's LLVM backend) and CUDA from the same code** — the device is picked at runtime,
so there's no separate CPU fallback to maintain. For solver-only use without the test
stack, install `.[solver]`.

```bash
python -m lumen.hardware     # -> {"device": "cuda"|"cpu", ...}
```

Backend chatter is quiet by default so example output stays parseable. Set
`LUMEN_BACKEND_LOG_LEVEL=info` (or `debug`) when you want Warp/Newton diagnostics.

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

The geometry core works standalone, no solver backend required:

```python
from lumen.core.frame import CenterlineFrame
hit = CenterlineFrame(pts).project(np.array([1.0, 0.0, 50.0]))   # -> (s, θ, r)
```

## The benchmark rewards safe navigation, not just reach

A policy that reaches the target by scraping along the vessel wall has not solved a
medical task. So the benchmark scores **safe** reach (target reached without crossing
the wall-safety threshold) separately from raw reach, and the leaderboard ranks safe
success first.

<p align="center">
  <img src="docs/assets/benchmark.svg" alt="Safe vs. unsafe success across navigation tasks" width="820">
</p>

```text
task               tier        safe   unsafe  success  mean_steps   max_pen
nav_tube           easy        1.00     0.00     1.00        18.6     0.000
nav_stenotic       medium      1.00     0.00     1.00        19.8     0.000
nav_tree_branch    hard        0.00     1.00     1.00        51.0     1.325

overall: safe=0.67  unsafe=0.33  success=1.00  worst max_pen=1.325  mean_return=60.6
```

The forward baseline reaches every target, but only breaches the wall on the hard
branch — which is exactly the case a real policy has to learn to do cleanly. Run it
and submit your own policy:

```bash
lumen benchmark /tmp/lumen-bench
python examples/submit_policy.py /tmp/lumen-bench my-lab-policy   # copy, replace policy(obs), score
python examples/train_fluoro_nav.py                              # train from images, not privileged state
```

Rejected scorecards print the schema or comparability reason instead of failing
silently, so a malformed submission never quietly disappears from the leaderboard.

## From a solver step to a training batch

The same toolchain takes a simulated intervention all the way to a fixed-shape tensor
batch, and refuses bad data at every stage — so a training job fails at the index, not
halfway through an epoch.

<p align="center">
  <img src="docs/assets/pipeline.svg" alt="lumen data pipeline: solver to training batch" width="880">
</p>

```bash
lumen capture           /tmp/lumen-episodes                    # replayable case bundles + previews
lumen validate          /tmp/lumen-episodes --require-cv-labels
lumen index             /tmp/lumen-episodes --out /tmp/lumen-episodes/index.jsonl --modality fluoro
lumen inspect-index     /tmp/lumen-episodes/index.jsonl --check-arrays --require-cv-labels
lumen materialize-batch /tmp/lumen-episodes/index.jsonl /tmp/lumen-episodes/batch.npz --limit 32
```

- **`capture`** writes one self-contained directory per scenario, each with a
  `preview.png`, fluoro device/vessel mask contact sheets, and a label overlay — so you
  can inspect observations and CV labels without opening a single NumPy sidecar.
  `lumen.data.rollout_episode(..., policy_observation="image")` lets capture/training
  policies receive rendered fluoro or luminal observations instead of the default fast
  privileged 5-D state observation; stored image-policy steps reuse that same pre-action
  frame so behavioral-cloning pairs align `step.obs` with `step.action`.
- **`validate`** checks every bundle's asset, calibration, observations, masks,
  keypoints, and labels before you train on it. `--require-cv-labels` makes masks and
  tip/base keypoints mandatory on every fluoro frame.
- **`index`** writes one JSONL row per timestep (observation, masks, node positions,
  keypoints, action, clinical metrics, calibration, provenance), with paths relative to
  the index so sibling outputs stay portable.
- **`inspect-index`** summarizes the corpus and, with `--check-arrays`, loads the
  referenced arrays to report shapes, mask coverage, and keypoint-to-device distances —
  and rejects empty masks or off-frame keypoints before a training job opens anything.
- **`materialize-batch`** writes a compressed `.npz` smoke-test batch plus a manifest,
  failing fast on missing or mixed-shape arrays so you can test tensor ingestion before
  a full dataloader run.

For a minimal NumPy dataloader over the index:
`python examples/load_fluoro_index.py /tmp/lumen-episodes/index.jsonl --limit 8`. Every
`lumen` subcommand is also installed as a standalone `lumen-*` script for shell
pipelines. See [docs/EPISODE_SCHEMA.md](docs/EPISODE_SCHEMA.md) for the on-disk format.

## What's inside (`lumen.newton`)

| Piece | What it does |
|---|---|
| **Guidewire** | Newton `add_rod` cable — stretch + bend/twist; torsion carries proximal rotation to the tip |
| **`TubeVBDSolver`** | a fork of Newton's `SolverVBD` that injects the tube-intrinsic contact **barrier + Hessian** into the AVBD solve, so contact stays implicit and stable |
| **HGO wall** | Holzapfel–Gasser–Ogden anisotropic shell as the deformable lumen field `R(s,θ)=R0+w` |
| **Anisotropic friction** | Coulomb friction with μ varying by slide angle to the collagen fiber direction |
| **Clot** | finite-extent Ogden occlusion that collapses the shared `R`; progressive damage → fragmentation; stent-retriever capture/slip/fragment |
| **Flow** | 1-D resistive pressure field `P(s)` / `v(s)` along the centerline (clot raises resistance, aspiration is a pressure sink), with a lumped Windkessel fallback |
| **Cross-validation** | fast-tier kernels checked against analytic ground truth to ~1e-6; STARK / ppf-contact-solver drop-in slot |

There are two solver tiers: a **fast tier** (`lumen.newton`, batched Newton VBD) built
for RL throughput, and an **accurate tier** (`lumen.accurate`) with a self-contained
penetration-free IPC reference and Warp-autodiff gradients for offline calibration. The
fast tier's force→indentation response is cross-validated against the accurate tier on
the same scene. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design invariants.

## Layout

```
lumen/core/       frame · lumen_field        tube-intrinsic geometry (NumPy only)
lumen/newton/     sim · tube_vbd · tube_barrier_kernel · hgo_wall · clot · flow · devices · crossval
lumen/accurate/   ipc · diff · stochastic    cross-validation, gradients, sysID
lumen/assets/     schema (the private-data seam) · procedural generator
lumen/data/       capture · replay · index · calibrate    the episode toolchain
lumen/profiles/   endovascular | …           the repurposing surface
lumen/envs/       NavEnv (Gym, Newton-backed)
lumen/hardware.py device detection (cuda/cpu)
tools/            firewall check (no CathSim, no patient data)
```

## License & boundaries

[Apache-2.0](LICENSE), deliberately **clean-room**:

- **No [CathSim](https://github.com/robotvisionlabs/cathsim)** — its CC-BY-NC-SA-4.0
  license would contaminate this one.
- **No patient data** — every committed asset is procedurally generated
  (`provenance="procedural"`).

Both are enforced in CI by `tools/check_firewall.py`. Real-data calibration and patient
pipelines live in the private Seldinger repos, behind the `lumen.assets.schema` and
`lumen.data.schema` seams, and layer on top of this open core — they never land here.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). In short: sign your
commits off (`git commit -s`, [DCO](https://developercertificate.org/)), keep `pytest`
and the firewall green, and open a PR. A new modality is a new directory under
`lumen/profiles/` — the core never changes to accommodate one.
