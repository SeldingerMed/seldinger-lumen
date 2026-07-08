---
title: lumen
---

# lumen

**A differentiable, GPU-parallel physics simulator for AI in a deformable tube.**

A guidewire in a vessel, a scope in an airway, an endoscope in a bowel — the same
physics: a slender device threading a soft, moving tube. `lumen` solves that one
problem and stays modality-agnostic. It's built to be a better base for learning
endovascular/intraluminal control than existing options like
[CathSim](https://github.com/robotvisionlabs/cathsim) — a genuinely deformable wall
instead of a rigid pipe, a safety-scored benchmark, and CV-ready data — and runs *on*
the [NVIDIA Newton](https://github.com/newton-physics/newton) engine.

<p align="center">
  <img src="assets/demo/nav_bifurcation.gif" alt="Guidewire navigating a branching vessel into the target branch" width="330">
  <img src="assets/demo/fluoro_bifurcation.gif" alt="Synthetic fluoroscopy of the same branching-vessel navigation" width="330">
  <br>
  <em>A guidewire navigating a branching vessel — entering the target branch at the fork
  and reaching the target, solved on Newton. Left: <strong>schematic</strong> (wire cyan,
  target gold). Right: synthetic <strong>fluoroscopy</strong>, what an ML model sees.
  Rendered with <code>lumen play</code>.</em>
</p>

[View on GitHub](https://github.com/SeldingerMed/seldinger-lumen){: .btn }

## Install

```bash
pip install -e ".[dev]"
```

`.[dev]` includes tests, Gymnasium, Warp, and the pinned Newton commit this solver is
validated against. For runtime-only solver use, install `.[solver]` instead. Runs on
**CPU and CUDA** from the same code (Warp picks the device at runtime).
Set `LUMEN_BACKEND_LOG_LEVEL=info` or `debug` to show Warp/Newton backend diagnostics.
Run `lumen doctor` when bringing up a new workstation or CI runner; it reports the
installed Lumen/Warp/Newton versions, pinned-backend validation status, CUDA visibility,
and actionable reinstall guidance when optional solver dependencies are missing.

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

## First 10 minutes for RL/CV users

```bash
lumen hardware
lumen play stenotic --out /tmp/lumen-run
lumen train tube --out /tmp/policy.npz
lumen play tube --policy /tmp/policy.npz
lumen benchmark /tmp/lumen-bench
lumen render-fluoro /tmp/lumen_fluoro.png
lumen capture /tmp/lumen-episodes
lumen validate /tmp/lumen-episodes
lumen replay /tmp/lumen-episodes
lumen index /tmp/lumen-episodes --out /tmp/lumen-episodes/index.jsonl --check-sidecars
lumen inspect-index /tmp/lumen-episodes/index.jsonl --check-arrays --require-cv-labels
lumen materialize-batch /tmp/lumen-episodes/index.jsonl /tmp/lumen-episodes/smoke_batch.npz --limit 32
lumen split-index /tmp/lumen-episodes/index.jsonl --out-dir /tmp/lumen-episodes/splits
lumen calibrate
```

`capture_episode.py` writes replayable case bundles with `preview.png`,
`preview_contact_sheet.png`, fluoro device/vessel mask contact sheets, and
`label_overlay_contact_sheet.png`. `lumen validate` checks every bundle's asset,
calibration, observations, masks, keypoints, labels, and sidecar refs before you
train on it; add `--require-cv-labels` when a fluoro CV run must have
device/vessel masks and tip/base keypoints on every frame. The replay summary reports clinical flags plus
annotation coverage such as
`device_mask=19/19`, `vessel_mask=19/19`, and
`keypoints(base=18/19 tip=19/19 nodes=170/171)`, so a CV pipeline can screen
masks/keypoints before loading arrays. `lumen index` writes a
JSONL dataloader index with observation, mask, node-position, keypoint, action,
clinical-metric, label, calibration, and provenance fields. Paths are
relative to the index file by default, so sibling or nested index outputs can be
loaded with `iter_index_records(path, load_arrays=True)`; pass
`--absolute-paths` for a machine-local index. Pass `--modality fluoro
--require-cv-labels` to write a fluoro-only training index that fails on missing
or empty CV labels. `lumen inspect-index --check-paths` summarizes rows,
modalities, labels, calibration types, episode-level clinical outcome/safety counts,
keypoint coverage, and missing sidecar references before a training job opens arrays;
add `--require-cv-labels` to fail if fluoro rows lack mask refs or present tip/base
keypoints, `--check-arrays` to load referenced arrays, report observation/mask/node
shape and dtype counts, report mask coverage and keypoint-to-device distances,
reject empty/bad masks, and catch off-frame or off-device keypoints, and add
`--json` for scripts and notebooks. Add `--require-uniform-arrays` before
fixed-shape batch training to fail if any loaded array field mixes shape/dtype
payloads. `lumen materialize-batch` turns an inspected JSONL index into a strict
compressed `.npz` smoke-test batch plus `.manifest.json`; it refuses missing or
mixed-shape requested arrays so CV/RL jobs can test tensor ingestion before a
full training run. Use
`--keypoint-mask-tolerance` to tune how far device
keypoints may sit from the device mask before the index fails. For
training loops, `CaseBundle.load(path).replay(include_annotations=True)` yields
each observation with lazy-loaded annotation arrays. `lumen split-index` writes
episode-grouped `train.jsonl`, `val.jsonl`, `test.jsonl`, and `manifest.json` files
from a validated index so procedure frames cannot leak across ML folds; use
`--seed`, `--ratios`, and `--stratify` to reproduce split assignments. The splitter
preserves sidecar paths from the source index as-is, so when splitting to a different
output directory the source index must be created with `--absolute-paths`, or keep
split outputs alongside the index to maintain relative path validity for array loading.
For a minimal NumPy dataloader-style batch, run
`python examples/load_fluoro_index.py /tmp/lumen-episodes/index.jsonl --limit 8`.
The same tolerance option is available on `lumen validate` and `lumen index`
when `--require-cv-labels` is enabled, so bad device labels can be stopped before
writing an index.

The standalone `lumen-*` scripts, including `lumen-validate`, remain installed for
shell pipelines.

The benchmark separates raw target reach from clinically safe reach:
`safe_success_rate` is target reach without wall-safety breach, while
`unsafe_success_rate` is target reach that required a safety breach. The leaderboard
ranks safe success before raw success, then lower wall penetration, then return.

Calibration uses wall-probe episodes, not navigation rollouts:
`examples/calibrate_from_episode.py` shows the biplanar identifiability check and
`lumen.data.probe_episode(...)` creates the replayable probe. Use
`lumen.data.joint_probe_episode(...)` when you need the wall+friction calibration
seam (`C10` and `mu`) instead of stiffness alone.

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

Apache-2.0 · every asset is procedurally generated, so use it freely.
