# Lumen vs CathSim: head-to-head protocol (X series, item 8)

## What is compared
- **Task class:** branch / arch navigation. Lumen `Lumen/NavTreeBranch-v0` (procedural bifurcation, 50 mm trunk, 50 mm branches, 2 mm radius, 45 degrees, target = left branch outlet). CathSim `phantom3` aortic-arch phantom, target = brachiocephalic artery (CathSim's default task). The anatomies differ; the task class (steer a wire into a chosen branch) is the same, and each simulator is run on its own standard task.
- **Learner:** Stable-Baselines3 PPO with identical settings in both: SB3 defaults plus a [64, 64] MLP trunk, n_steps 2048, batch 64. 600,000 environment steps per training run. MLP policy on Lumen's state vector; CathSim's MultiInputPolicy on its joint-state observation (`use_pixels=False`).
- **Training seeds:** 0, 1 and 5 for each simulator (amended 2026-11-21 from 0 to 5 to fit CathSim's ~15 h per seed on CPU; the set is the three CathSim seeds already in training, fixed before any CathSim result existed). Lumen results for all six trained seeds are kept in the JSON as `lumen_all_trained_seeds`. Every compared seed is reported, including failures. No reruns, no seed selection by outcome.
- **Evaluation:** each trained policy plays the same 100 frozen evaluation seeds (10000 to 10099), deterministic actions, 300-step cap. For CathSim, Python `random`, NumPy, and dm-control Composer / task RNGs are seeded per evaluation seed alongside `env.reset(seed=...)` so initial guidewire pose and observation noise are deterministic per seed. Lumen's `NavTreeBranch-v0` (`TreeNavEnv.reset()`) is deterministic (it ignores `seed` and resets to the same initial vessel/wire pose), so the 100 evaluation episodes repeat the same case on a deterministic policy; we retain the 100-seed harness for structural parity and report `distinct_eval_outcomes` in the JSON.
- **Statistics:** the training seed is the unit (n = 3 per simulator). Means with two-sided 95% t-intervals across training seeds; rates clipped to [0, 100%].
- **Hardware:** one Apple M4 Max (Mac Studio), CPU only, one environment per training run, single-threaded BLAS, same machine for both.
- **Versions:** Lumen at the commit in `manifest.json`; CathSim `airvlab/cathsim` main @ `adfabb2`, MuJoCo 2.3.7, dm-control 1.0.14, SB3 2.4.0 (CathSim's supported stack). Lumen: Newton/Warp 1.14, SB3 2.9.2.

## Metrics
- **Raw success:** tip within the task's tolerance of the target (Lumen 2.5 mm; CathSim 4 mm, CathSim's own `delta`).
- **Safe success:** success without exceeding the simulator's stated safety limit:
  - **CathSim: peak tip-wall contact force 1.0 N** (sensitivity 0.5 N and 1.5 N). Force = MuJoCo per-contact normal force on guidewire-tip/phantom contacts, Newtons, CathSim's own `get_contact_forces` definition (contacts within 2 mm, above 1 mN), sampled every 12 ms control step.
  - **Lumen: device-wall surface overlap 0.3 mm**, equal to the contact barrier width `d_hat`. Lumen reports wall load in simulator units, not Newtons, so its limit is geometric.
  - The two limits are not cross-calibrated, and the comparison does not equate them.
- **Peak and mean tip force (N):** CathSim. Lumen's wall load is reported in simulator units.
- **Time to target:** simulated seconds over successful episodes (Lumen 20 ms per step; CathSim 12 ms per step).
- **Wall-clock and throughput:** training hours for 600k steps; environment steps per second in training and evaluation.
- **CathSim on its own published metrics** (`cathsim/rl/metrics.py`): root constraint norm mean/max, safety (share of steps with norm at or below 2 on CathSim's `info["forces"]`), episode length, path length, success (episode under `max_steps`), per-episode SPL.

## Why 1.0 N
Typical guidewire tip contact forces during competent navigation are about 0.05 to 0.3 N (intraluminal force monitoring in supra-aortic vessels, arXiv:2012.10762; sensorized vessel platforms, Int J CARS 2023). Catheter-vessel tribology work treats forces above 1 N as critical for endothelial damage (Takashima et al., Tribology Int 2007, via the Leeds bio-tribology review). The 1.5 N figure common in the RL-navigation literature (Jackson et al., Int J CARS 2023) is a safety-factor estimate, ten times a 0.15 N rabbit-vein needle puncture, not a measured rupture force; we report it as a sensitivity bound. 0.5 N is the strict bound, matching finite-element navigation forces (Oussalah et al., Int J CARS 2025) and CathSim's reported maximum of about 0.26 N.

## Implementation note
CathSim's gym wrapper calls `get_contact_forces(self.physics, self.image_size)`, which passes the image size (64) into the force `threshold` argument, so `info["contact_forces"]` drops every contact below 64 N. The evaluation script reads contact forces from a copied MuJoCo physics state (so probing never mutates live `qacc_warmstart` or rollout trajectories) with CathSim's documented 1 mN threshold instead. CathSim's `info["forces"]` is `||qfrc_constraint[0:3]||` across the guidewire root slide and hinge DOFs (mixed linear-force and torque generalized constraint norm, not pure Newtons), so it is used only for CathSim's native metrics.

## Reproduce
```
# train + evaluate (both simulators)
# one process per training seed s in 0 1 5 (the compared set), OMP/MKL/VECLIB threads = 1
# Lumen runs in this repo's environment; CathSim is installed separately in its own venv
# (airvlab/cathsim @ adfabb2, CC-BY-NC-SA-4.0) and is never vendored into this repo.
python benchmarks/external_comparison/train_sb3.py --environment lumen   --task nav_tree_branch --algo ppo --timesteps 600000 --eval-episodes 100 --seeds $s --run-id lumen-nav_tree_branch-ppo-main-s$s --out-dir BENCH_DIR --model-dir MODELS_DIR
MUJOCO_GL=disable python benchmarks/external_comparison/train_sb3.py --environment cathsim --task phantom3_bca --algo ppo --timesteps 600000 --eval-episodes 100 --seeds $s --run-id cathsim-phantom3_bca-ppo-main-s$s --out-dir BENCH_DIR --model-dir MODELS_DIR --checkpoint-every 25000
# CathSim tip-contact force on the frozen models
MUJOCO_GL=disable python benchmarks/external_comparison/eval_comparator_contact.py --models MODELS_DIR --seeds 0,1,5 --out cathsim-contact.json
# table, chart, matched episodes, video
python benchmarks/external_comparison/xseries_compare.py --bench BENCH_DIR --contact cathsim-contact.json --seeds 0,1,5 --out compare
python benchmarks/external_comparison/xseries_chart.py --compare compare.json --out chart.png
python benchmarks/external_comparison/record_matched.py --environment lumen   --models MODELS_DIR --seeds 0,1,5 --out episodes
python benchmarks/external_comparison/record_matched.py --environment cathsim --models MODELS_DIR --seeds 0,1,5 --out episodes
python -m tools.xseries.clip_matched --episodes episodes --compare compare.json --out VIDEO_DIR
```
