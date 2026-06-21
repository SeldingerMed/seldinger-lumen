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
sensor. Profiles (`lumen/profiles/<x>/`) bundle the choice; sensors
(`lumen/sensors/`) provide the observation. Adding bronchoscopy must touch
neither `lumen.core` nor existing profiles.

## Invariant 3 — the shared `R` field

Wall mechanics and contact geometry are the **same object** `R(s,θ,t)` (doc
§3.5.6). Wall deformation = a change in `R`; the contact barrier reads the same
`R`; pulsatility = a temporal modulation of `R`. Do not introduce a separate
collision mesh that can drift out of sync with the wall state.

## Invariant 4 — two tiers

- **Fast tier** (this repo, Warp/Newton): batched, differentiable-with-care,
  reduced-order shell. For RL throughput.
- **Accurate tier** (external oracle: STARK/SymX, ppf-contact-solver):
  cross-validation and offline calibration only. We *consume* it; we don't
  reimplement IPC. Hooks live in `lumen/core/validate/` (P2+).

## Invariant 5 — the open/closed firewall

Apache-2.0 clean-room. Enforced by `tools/check_firewall.py` in CI:

1. **No CathSim** (CC-BY-NC-SA-4.0) anywhere — it would contaminate the license.
2. **No patient data** — every committed asset is `provenance="procedural"`.

The seam to the private world is `lumen.assets.schema.Asset`. A patient pipeline
in `seldinger-ml` emits that schema (with `provenance="patient(private)"`) and
keeps it out of this repo. Real-data HGO calibration, clot models, the GNN flow
surrogate, and trained policies all stay private and layer *on top of* this core.

## Coordinate frames

Assets declare their frame explicitly (`Frame.name/spacing_mm/origin_mm`).
Default is `voxel_scaled` (voxel × spacing, origin 0) to match the existing
Seldinger route format. The solver is frame-agnostic but frame-*explicit*: it
never assumes; it reads what the asset declares.
