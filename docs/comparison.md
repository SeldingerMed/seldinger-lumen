# Lumen vs. other open endovascular simulators

Lumen is a clean-room, Apache-2.0 project. It does **not** vendor, import, or derive
from CathSim (CC-BY-NC-SA), and a CI firewall (`tools/check_firewall.py`) enforces
that. This page compares *capabilities* — observed feature surface — so users can see
where Lumen stands. It is not a port and shares no code.

The reference point is CathSim, the most-cited open endovascular RL simulator
(MuJoCo / `dm_control`, rigid aorta, force sensing, SB3 agents).

> **How this was checked.** CathSim was installed in an isolated environment and
> *run* — `gymnasium.make("cathsim/CathSim-v0")`, reset, and stepped — to observe its
> real runtime surface, not just its source. Observed: `action_space = Box(-1, 1, (2,))`
> (advance + rotate), `obs = {joint_pos: (168,), joint_vel: (168,)}`, dense reward. Its
> interactive `run_env` viewer and pixel/segment observations need an on-screen GL
> context (GLFW/EGL), which a headless box does not provide — the same constraint
> Lumen sidesteps with its headless schematic viewer.

## Feature parity

| Capability | CathSim | Lumen |
|---|---|---|
| Watch a rollout (interactive/instant) | `run_env` MuJoCo viewer | **`lumen play`** — headless schematic animation (`.avi`+`.png`), CPU-only |
| Train a policy from the CLI | `train` (SB3 SAC/PPO) | **`lumen train`** — gradient-free CEM, no torch |
| Visualize a trained agent | `visualize_agent` | **`lumen play --policy <npz>`** |
| Gym / Gymnasium env | `gym.make("cathsim/CathSim-v0")` | `gymnasium.make("Lumen/NavTube-v0" \| "…/NavStenotic-v0" \| "…/NavTreeBranch-v0")` |
| Action space | 2-DoF `Box(-1,1,(2,))` — advance + rotate | 1-DoF insertion (the sim supports twist via `sim.step(twist=…)`; see note below) |
| Proprioceptive obs | `joint_pos` + `joint_vel`, 168-D each | compact 5-D tip-centric state `[s/L, r/R, sinθ, cosθ, (target−s)/L]` |
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

## Known gap: the rotation action

Running CathSim surfaced one real control-surface difference: its action is 2-DoF
(**advance + rotate**), while Lumen's navigation envs expose only insertion. Lumen's
solver *does* actuate proximal twist (`sim.step(twist=…)`, torsion carried to the tip),
so the physics is there. But on Lumen's straight, symmetric procedural vessels,
spinning a straight guidewire is physically inert — measured tip/node displacement
between `twist=0` and `twist=1` is ~1e-5. Rotation only earns its place with a
pre-shaped/J-tip device or a branch-torque steering task, where a twist swings the
tip to select a branch. Exposing a 2nd action dimension that does nothing on the
current scenes would be hollow parity, so it is **scoped as a follow-up** (angled-tip
device option + a torque-to-select-branch task) rather than shipped empty.

## Deliberate non-goals in the open core

- **Live keyboard tele-operation.** `lumen play` is policy-driven and headless by
  design (CI/servers/parallel agents). A live interactive viewer can layer on top.
- **Patient meshes / mesh-processing pipeline.** Real anatomy lives behind the private
  asset-schema seam; the open repo ships only procedurally generated geometry.
- **VR/AR front-ends.** Out of scope for Layer 0.
