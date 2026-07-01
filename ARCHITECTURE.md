# Architecture

`lumen` is the open-core Layer 0 of the Seldinger stack: a domain-agnostic,
differentiable, GPU-parallel solver for *a continuum instrument in a deformable
lumen, observed through a sensor*. This file records the design invariants so the
generality survives contact with feature work.

## Invariant 1 — the core names no anatomy

`lumen.core.*` must never reference vessels, airways, bowel, etc. It manipulates:

- a **centerline** → a tube-intrinsic frame `(s, θ, r)` (`frame.py`)
- a **lumen field** `R(s, θ, t)` shared by wall + contact (`lumen_field.py`)
- a **continuum instrument** (Cosserat/VBD rod + torsion) — *P1*
- a **deformable wall** (reduced anisotropic shell that deforms `R`) — *P2*
- **contact** (tube-intrinsic narrowphase + analytic barrier, Warp) — *P1*
- **coupling** (flow / fluid fields) — *P4*
- a **solver** (Newton custom solver, batched envs) — *P1/P2*

Anything anatomy-specific is a *profile* or a *sensor*, not core.

## Invariant 2 — three swap points, no core change

Repurposing across procedures (doc §3.9) = swapping anatomy field + instrument +
sensor. Profiles (`lumen/profiles/<x>/`) bundle the choice; the sensor (the
observation model, Layer 1) is future work. Adding bronchoscopy must touch neither
`lumen.core` nor existing profiles.

## Invariant 3 — the shared `R` field

Wall mechanics and contact geometry are the **same object** `R(s,θ,t)` (doc
§3.5.6). Wall deformation = a change in `R`; the contact barrier reads the same
`R`; pulsatility = a temporal modulation of `R`. Do not introduce a separate
collision mesh that can drift out of sync with the wall state.

## Invariant 3b — one engine (Newton), CPU and GPU; no parallel engine

The solver is the Newton engine, which runs on the CPU (Warp's LLVM backend) and
on CUDA — the same code, device chosen by `lumen.hardware.detect_device()`. There
is **no** separate PyTorch engine: Newton already covers CPU, and a parallel
reduced-order engine would violate "do not write an engine" (§3.2). (An earlier
torch path existed as a prototype and was removed once Newton-on-CPU was
confirmed.) The contact barrier is a Warp kernel injected into VBD's AVBD solve
(`newton/tube_barrier_kernel.py` + `newton/tube_vbd.py`, a thin `SolverVBD`
subclass overriding only the per-color rigid-body iteration).

## Invariant 4 — two tiers

- **Fast tier** (`lumen/newton`, Newton VBD): batched, for RL throughput. Newton
  VBD is not autograd-differentiable; that is by design (doc §3.5.7).
  - **Throughput.** The bible's target (§3.3) is `>=1e4` aggregate env-steps/s on a
    workstation GPU for single-device navigation. The batched step is round-trip-free
    (~97% Newton-VBD-kernel time; per-substep co-sim runs through on-device kernels,
    no host sync), so per-env cost falls sharply with batch size — verify with
    `python examples/benchmark_throughput.py --device cuda --require-cuda`. The
    `.github/workflows/gpu-benchmark.yml` workflow is the scheduled/manual hardware
    gate for that claim on a self-hosted CUDA runner; it records `python -m
    lumen.hardware`, runs the Newton/Warp kernel tests, runs the CUDA throughput
    benchmark with `--min-env-steps-per-s 10000`, and uploads JSON artifacts.
    `test_throughput` pins that amortization on CPU as a regression guard: a stray
    `.numpy()` in the hot loop (a device→host sync) would serialize the envs and is
    what this catches.
- **Accurate tier** (`lumen/accurate`): cross-validation, penetration-free IPC, and
  gradients for offline calibration.
  - `accurate/ipc.py` — a *built-in, self-contained* penetration-free IPC reference
    (quasi-static rod + log barrier + feasibility line search). Always available, no
    GPU/C++ build; it's a reference, not a production engine.
  - `accurate/diff.py` — the differentiable calibration path (Warp autodiff), fitting
    constitutive parameters to observations.
  - `accurate/stochastic.py` — physically-grounded stochastic contact gradients (§3.5.8):
    where the deterministic barrier gradient is dead (zero outside the active band), model
    the gap as jittered by physical uncertainty (blood film, roughness, manufacturing) so
    `E[reaction]` is smooth and the randomized-smoothing gradient is recoverable. A sysID
    tool (not a policy claim); `sigma` trades reach for bias — the §3.5.8 option, evaluated.
  - `crossval.crossval_indentation_response` — the oracle ROLLOUT check: the fast tier's
    force→indentation response is validated against the penetration-free IPC oracle on the
    same scene (a discretisation-robust scalar per load, not a node-wise shape match).
  - The heavy external oracles (STARK/SymX, ppf-contact-solver) remain a drop-in via
    the same `crossval.accurate_tier_status` seam for those who build them on a GPU
    box — we don't reimplement *production* IPC, only a reference.

## Invariant 4b — support matrix is part of the solver contract

The fast tier is batched for the core guidewire/tube contact path, but several
combined physics modes remain intentionally single-env. The public contract lives in
[docs/SOLVER_SUPPORT.md](docs/SOLVER_SUPPORT.md), which maps each single-env vs.
`n_envs > 1` path to the exact runtime guard and follow-up issue. Any change that
removes a `NotImplementedError` from `lumen/newton/sim.py` must update that matrix and
add a regression test for the newly supported combination.

## Invariant 5 — the open/closed firewall

Apache-2.0 clean-room. Enforced by `tools/check_firewall.py` in CI:

1. **No CathSim** (CC-BY-NC-SA-4.0) anywhere — it would contaminate the license.
2. **No patient data** — every committed asset is `provenance="procedural"`.

There are now two seams to the private world, both firewall-guarded:
`lumen.assets.schema.Asset` (geometry, `lumen-asset/0`) and
`lumen.data.schema.Episode` (a captured intervention — kinematics + paired
observation + outcome, `lumen-episode/0`). A patient pipeline in `seldinger-ml`
emits both (with `provenance="patient(private)"`) and keeps them out of this repo.
Real-data HGO calibration, clot models, the GNN flow surrogate, and trained
policies all stay private and layer *on top of* this core.

## Invariant 6 — Layer 2 ships the standard + machinery, not the corpus

Layer 2's *value* is the paired real-data corpus — that is the proprietary moat
and stays private (§5). The open repo ships the **data standard** (`Episode`) and
the **machinery** proven end-to-end on procedural data: synthetic capture
(`lumen.data.capture`), corpus iteration/replay (`lumen.data.replay`), and the
sim2sim calibration loop (`lumen.data.calibrate`, which closes §3.6 — recover wall
stiffness from an episode's fluoro, error-checked against the stored ground truth).
Real patient capture plugs into the *same* `Episode` seam; nothing about it lives
here. External open datasets (Guide3D et al., §4.2) are a documented future
adapter seam — licensing/firewall-gated, not vendored.

## Coordinate frames

Assets declare their frame explicitly (`Frame.name/spacing_mm/origin_mm`).
Default is `voxel_scaled` (voxel × spacing, origin 0) to match the existing
Seldinger route format. The solver is frame-agnostic but frame-*explicit*: it
never assumes; it reads what the asset declares.
