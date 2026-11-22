# Lumen vs CathSim v2: identical anatomy, time to success (preregistration)

Status: preregistered 2026-11-22, before any v2 run. Changes after this commit are listed under
Amendments with a date and reason; no change is made in response to a result.

## Anatomy (identical in both simulators)
- Geometry: CathSim `phantom3` aortic-arch phantom, read at runtime from a separate CathSim install
  (`phantom_assets/meshes/phantom3/visual.stl`, MuJoCo world frame and scale). Nothing from CathSim
  is committed to this repository.
- Lumen import: the visual surface mesh is voxelized (0.25 mm), its lumen centerline graph is
  extracted, and each edge gets a cross-section radius field R(s, theta) from the mesh. The
  conversion error (max and RMS distance between the imported Lumen wall and the source mesh) is
  measured and reported in the manifest before training starts.
- Start pose: CathSim's guidewire insertion point and direction, transformed into the same frame.
- Target: CathSim's target sites for `bca` and `lcca`; success = tip within 4 mm of the target
  (CathSim's `delta`) in both simulators. Episode cap 300 control steps in both.

## Tasks
- Standard: target BCA (CathSim's default task).
- Harder variant (fixed now): target LCCA, CathSim's second supported target. It is the second
  branch off the arch, so the wire must be steered past the BCA ostium before turning into the
  target branch. Same tolerance, cap and settings as BCA. No other variant is run.

## Learner (identical settings)
Stable-Baselines3 PPO, SB3 defaults plus a [64, 64] MLP trunk, batch 64, 600,000 environment steps
per run, total rollout of 2,048 transitions per update (n_steps x n_envs = 2048). Training seeds 0, 1
and 5 in each simulator and configuration.

## Configurations
1. Single environment vs single environment.
2. Best configuration of each simulator on this machine: Lumen batched (E worlds in one Newton model);
   CathSim SB3 `SubprocVecEnv` with n worker processes. E and n are chosen by a fixed 120 s
   throughput probe (highest env-steps/s) before any training run, and recorded. CathSim worker count
   is bounded by the cores and memory admitted to it by the machine's admission queue; the admitted
   envelope is recorded.

## Headline metric: wall-clock time to 90% success
- A checkpoint is saved every 25,000 steps. Each checkpoint is evaluated on the same 100 frozen
  evaluation seeds (10000 to 10099), deterministic actions.
- Steps to target = steps at the first checkpoint with success >= 90%. Wall-clock to target = steps
  to target / that run's measured training throughput (env-steps/s over uninterrupted training
  segments), so pauses imposed by the shared machine do not count for either simulator.
- A run that never reaches 90% within 600,000 steps is reported as not reached, with its best
  success; it is not imputed.
- Reported per training seed and as mean with a two-sided 95% t-interval across seeds.
- Throughput (env-steps/s) is reported for training and evaluation in both configurations.

## Safety readouts (each on its own scale, never on a shared axis)
- CathSim: peak and mean guidewire-tip contact force in newtons from the MuJoCo contact state,
  against 1.0 N (rationale in XSERIES_PROTOCOL.md).
- Lumen: device-wall surface overlap (mm) and wall load (simulator units).
- No cross-simulator safe-success comparison is made.

## Hardware
One Apple M4 Max Mac Studio, CPU, same machine for every run; BLAS threads = 1 per worker.

## Reuse
The v1 CathSim BCA single-environment runs (seeds 0, 1, 5) used exactly these PPO settings and kept
every 25,000-step checkpoint. Their checkpoints are evaluated under this protocol instead of being
retrained. All other runs are new.

## Amendments
None.
