# Common Endovascular Benchmark

This directory holds the reproducible comparison harness for Lumen, CathSim, and
stEVE/SOFA-style environments. The contract is in `metric_contract.json`.

The benchmark compares matched task classes, not identical geometry. Pilot runs use
random, forward, and sweep policies to verify adapters and metric extraction. Main
paper-facing runs use PPO and SAC with CathSim-style budgets: 600,000 training steps,
six seeds, and 100 frozen-policy evaluation episodes per seed/task.

Safety is reported by native endpoint, not a shared `safe_success` field. Lumen
emits device-surface penetration and wall-load curves in simulator units; CathSim
emits its native contact-force curve; stEVE has no classified safety endpoint in this
harness. Cross-environment safety rates are intentionally withheld until matched
device/anatomy/material, solver-unit, and phantom or ex-vivo force--injury
calibration exists. Every preregistered run is retained regardless of outcome.

Run a preregistered multi-seed baseline explicitly:

```bash
python benchmarks/external_comparison/train_sb3.py \
  --environment lumen \
  --task nav_tree_branch \
  --algo ppo \
  --timesteps 600000 \
  --seeds 0,1,2,3,4,5 \
  --eval-episodes 100 \
  --run-id lumen-nav-tree-ppo-main
```

One result file contains every seed's model path, training time, episode rows, and
the aggregate seed schedule; no seed is silently replaced by an average.
Evaluation seeds use the same frozen block beginning at 10,000 for every model.
Each episode records `training_seed`, `model_id`, and evaluation `seed`; aggregate
rows stay per trained model instead of pooling independent runs.
If a training seed fails, its complete frozen evaluation block is emitted as failed
rows with the planned model identity, keeping the final artifact unconditional.
Run IDs are normalized to one safe filename component before model/result artifacts
are written.
Run records distinguish failure stage and total elapsed time from training time.

Example pilot commands:

```bash
python benchmarks/external_comparison/common_bench.py lumen \
  --episodes 30 \
  --policies random,forward,sweep \
  --run-id lumen-pilot

MUJOCO_GL=disable /path/to/cathsim-venv/bin/python \
  benchmarks/external_comparison/common_bench.py cathsim \
  --episodes 30 \
  --policies random,forward,sweep \
  --external-repo /path/to/cathsim \
  --run-id cathsim-pilot
```

stEVE/SOFA readiness is checked separately because stEVE requires SOFA, SofaPython3,
and BeamAdapter:

```bash
python benchmarks/external_comparison/common_bench.py smoke-steve --run-id steve-smoke
```

Combine aggregate rows after runs finish:

```bash
python benchmarks/external_comparison/summarize_results.py \
  benchmarks/external_comparison/results/lumen-pilot-30.json \
  benchmarks/external_comparison/results/cathsim-pilot-30.json
```
