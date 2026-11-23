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
1. 2026-11-22, before any v2 Lumen training or evaluation: geometry conversion as implemented in
   `phantom_import.py`.
   - Voxel size 1.0 mm instead of the planned 0.25 mm, to keep the importer under the shared
     machine's per-process memory cap. Voxels only seed the centerline graph; the radius field
     R(s, theta) is measured by exact ray casts against the source surface, so wall accuracy does
     not depend on voxel size (measured below).
   - Scale 1000 mm per MuJoCo metre; the visual geom's 1.005x display scale is not applied (it only
     separates the visual mesh from the collision hulls).
   - Shell mesh: the inner lumen surface is the smaller of the two large smooth patches after a
     55 degree crease split. Centerlines: cost-weighted (1/d^2) shortest paths from the inlet seed
     (0, 2.0, -0.5) mm to geodesically farthest uncovered lumen voxels; side openings are closed by
     periodic interpolation of R in theta. Edges are split at the CathSim target sites, which become
     the Lumen target nodes.
   - Conversion error, measured on 59,423 source-surface samples (open vessel end caps excluded) under the Newton tree-contact
     projection (the rule that produces wall forces): median 0.02 mm,
     RMS 0.76 mm, p95 1.40 mm,
     max 9.3 mm. Imported wall to source surface: median
     0.16 mm, p95 0.49 mm. Larger
     deviations concentrate within one vessel radius of branch points (p95
     3.2 mm there, 0.67 mm elsewhere).
     Target sites lie 0.56 mm (bca) and
     0.63 mm (lcca) from their centerlines; the inlet seed
     0.19 mm. Full report: `import_report.json` in the v2 manifest.
2. 2026-11-22, before any v2 checkpoint is scored: checkpoint evaluation stops early once the
   threshold is unreachable (`checkpoint_curve.py`). Seeds run in schedule order; after 11 failures a
   checkpoint cannot reach 90% and is recorded as below threshold, with the success over its evaluated
   prefix kept for the learning-curve plot and marked as partial. Every checkpoint that reaches 90% is
   scored on all 100 seeds, so the headline (first checkpoint at >= 90%) is unchanged. The snapshot at
   the full budget is replaced by the final saved model, because SB3 writes that snapshot before the
   last policy update.
3. 2026-11-23, before any v2 Lumen training and before any batched or multi-worker run: choice of the
   best configuration.
   - Admissible parallel counts divide gcd(2048, 25000) = 8, so every configuration keeps the same
     2,048-transition rollout per update and the same 25,000-step checkpoint grid as the single
     environment (the learning curves are scored at identical step counts). Candidates: 1, 2, 4, 8.
   - Probe: 20 s of uniformly random actions per configuration on phantom3 BCA, env-steps/s, run under
     the admission queue on this machine (the planned 120 s was shortened; the ranking is stable well
     within 20 s). The same probe script shape is used for both simulators.
   - Lumen (BatchedTreeNavEnv, CPU warp): E=1 69.8, E=4 118.9, E=8 134.6 env-steps/s (E=16 144.0 and
     E=32 149.8 were measured but are not admissible under the rule above). Chosen: E = 8.
   - CathSim: each SubprocVecEnv worker is a separate Python process of about 1 GB, and the shared
     machine policy admits at most 3 heavy processes, so subprocess parallelism is capped at 2 workers.
     Candidates: in-process DummyVecEnv with 4 and 8 environments, and SubprocVecEnv with 2 workers;
     the fastest is used. Measured values and the choice are appended below before any CathSim
     multi-environment run.
4. 2026-11-23, supplementary resolution check (declared after the Lumen single-environment learning curves
   were scored, so it is reported as supplementary and never replaces the preregistered metric).
   - Observation: every Lumen single-environment run (BCA and LCCA, seeds 0, 1, 5) was at >= 90% success at
     its first checkpoint (25,000 steps). Its preregistered time to 90% is therefore an upper bound set by the
     checkpoint grid, and is reported as such ("<=").
   - Check: the same Lumen configuration retrained from scratch for 25,000 steps per seed and target, with a
     snapshot every 2,500 steps, each snapshot scored by the same checkpoint_curve rule (100 frozen evaluation
     seeds, early stop after 11 failures). Run ids "fine". The first snapshot at >= 90% gives a point estimate
     within 2,500 steps.
   - Reporting: shown beside, never instead of, the preregistered 25,000-step-grid values.
