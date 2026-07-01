# Lumen vs. other open endovascular simulators

Lumen is a clean-room, Apache-2.0 project. It does **not** vendor, import, or derive
from CathSim (CC-BY-NC-SA), and a CI firewall (`tools/check_firewall.py`) enforces
that. This page compares *capabilities* — observed feature surface — so users can see
where Lumen stands. It is not a port and shares no code.

The reference point is CathSim, the most-cited open endovascular RL simulator
(MuJoCo / `dm_control`, rigid aorta, force sensing, SB3 agents).

## Feature parity

| Capability | CathSim | Lumen |
|---|---|---|
| Watch a rollout (interactive/instant) | `run_env` MuJoCo viewer | **`lumen play`** — headless schematic animation (`.avi`+`.png`), CPU-only |
| Train a policy from the CLI | `train` (SB3 SAC/PPO) | **`lumen train`** — gradient-free CEM, no torch |
| Visualize a trained agent | `visualize_agent` | **`lumen play --policy <npz>`** |
| Gym / Gymnasium env | `gym.make("cathsim/CathSim-v0")` | `gymnasium.make("Lumen/NavTube-v0" \| "…/NavStenotic-v0" \| "…/NavTreeBranch-v0")` |
| Image observations | pixels + segmentation masks | synthetic fluoroscopy + luminal RGB, **with pixel-exact vessel/device masks, tip/base keypoints, node positions** |
| Multiple anatomies | named phantoms (aorta meshes) | procedural `tube` / `stenotic` / `bifurcation` (+ private patient-asset seam) |
| Branch selection task | targets on one phantom | dedicated branching tree env (`NavTreeBranch`) |
| SB3 / external RL libraries | first-class | works via the Gymnasium env; CEM ships in-repo |

## Where Lumen goes further

- **Deformable, physiological wall.** The vessel is not a rigid pipe: it is a
  Holzapfel–Gasser–Ogden anisotropic hyperelastic shell, and wall mechanics and
  contact geometry are the *same* shared field `R(s,θ,t)` — they cannot desync.
- **Richer intraluminal physics.** Anisotropic fiber-aligned friction, instrument
  torsion, a finite-extent Ogden clot with progressive damage / stent-retriever
  capture, and a 1-D flow pressure field.
- **Safety-scored benchmark.** `safe_success_rate` (target reached *without* breaching
  a wall-penetration threshold) ranks above raw success. `lumen play` reports the same
  numbers, so a rollout that "succeeds" by perforating the wall is visibly unsafe.
- **CV-ready data pipeline.** `capture → validate → index → split → materialize-batch`,
  with a versioned `Episode` standard, deterministic leak-free train/val/test splits,
  and free segmentation/keypoint labels.
- **Differentiable calibration tier.** A penetration-free IPC reference plus Warp-autodiff
  parameter fitting, cross-validated against an analytic oracle to ~1e-6.
- **Modality-agnostic core + permissive license.** The same core serves airway/bowel
  scopes, and Apache-2.0 allows commercial and collaborative use that non-commercial
  licenses preclude.

## Deliberate non-goals in the open core

- **Live keyboard tele-operation.** `lumen play` is policy-driven and headless by
  design (CI/servers/parallel agents). A live interactive viewer can layer on top.
- **Patient meshes / mesh-processing pipeline.** Real anatomy lives behind the private
  asset-schema seam; the open repo ships only procedurally generated geometry.
- **VR/AR front-ends.** Out of scope for Layer 0.
