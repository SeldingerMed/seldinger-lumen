# External Comparison Run Log

## 2026-07-15

Comparator sources:

- Lumen: current repo commit from the active worktree.
- CathSim: `https://github.com/airvlab/cathsim`, commit `adfabb2f291e31e6d656e66d0052269f38db01bd`.
- stEVE: `https://github.com/lkarstensen/stEVE`, commit `f909e7122d846a3c2fec91eedddaa0ba88b6391d`.
- stEVE_bench: `https://github.com/lkarstensen/deve_bench`, commit `1c7640eca7ed3701d1651cedbfad8f5b71132ff2`.
- stEVE_training: `https://github.com/lkarstensen/stEVE_training`, commit `414534acf8d3c9c8bac050b86e38490573fd4693`.

Completed setup:

- Created `metric_contract.json` for pilot and main PPO/SAC comparison.
- Created `common_bench.py` for random/forward/sweep pilot episodes.
- Created `train_sb3.py` for PPO/SAC train/eval runs in a shared Python 3.10 environment.
- Created shared RL venv at `<external-workdir>/.venvs/common-rl`.
- Installed Lumen, CathSim, SB3, Torch, Warp, Newton, MuJoCo, and CathSim runtime dependencies in the shared RL venv.
- Verified `gym.make("Lumen/NavTube-v0")` and `gym.make("cathsim/CathSim-v0")` from the shared RL venv.
- Installed stEVE and stEVE_bench in `<external-workdir>/.venvs/steve`.
- Verified native stEVE Python layer imports but SOFA/SofaRuntime are absent.
- Fetched stEVE_training submodules over HTTPS for the Docker/SOFA route.
- Built native arm64 CPU Docker image `lumen-steve-cpu:20260715` for stEVE/SOFA after the upstream amd64/CUDA route failed under emulation.
- Verified stEVE/SOFA Docker execution with a patched VTK import (`vtkCapsuleSource -> vtkSphereSource`) needed by the available arm64 VTK wheel.

Completed result files:

- `results/lumen-smoke.json`
- `results/cathsim-smoke.json`
- `results/cathsim-smoke-reuse.json`
- `results/steve-smoke-local.json`
- `results/steve-smoke-venv.json`
- `results/lumen-pilot-30.json`
- `results/lumen-nav-tube-ppo-smoke-64.json`
- `results/cathsim-bca-ppo-smoke-64.json`
- `results/cathsim-pilot-30.json`
- `results/pilot-summary-lumen-cathsim.csv`
- `results/steve-pilot-basic-arch-30.json`
- `results/pilot-summary-lumen-cathsim-steve.csv`
- `results/lumen-tree-ppo-short-50k.json`
- `results/cathsim-bca-ppo-short-50k.json`
- `results/ppo-short-50k-lumen-cathsim-summary.csv`

Completed CathSim-30 pilot command:

```bash
MUJOCO_GL=disable <external-workdir>/.venvs/cathsim-py39/bin/python \
  benchmarks/external_comparison/common_bench.py cathsim \
  --episodes 30 \
  --max-steps 300 \
  --policies random,forward,sweep \
  --external-repo <external-workdir>/repos/cathsim \
  --run-id cathsim-pilot-30 \
  --progress
```

Completed stEVE Basic/Arch 30-episode pilot command:

```bash
docker run -i --rm \
  -v <repo>:/work \
  -w /work \
  lumen-steve-cpu:20260715 \
  python3 benchmarks/external_comparison/steve_pilot.py \
    --episodes 30 \
    --max-steps 150 \
    --tasks BasicWireNav,ArchVariety \
    --policies random,forward,sweep \
    --run-id steve-pilot-basic-arch-30 \
    --progress
```

Pilot result summary:

- Lumen forward policy: 100% success and 100% safe success on 5/5 native Lumen task classes.
- Lumen sweep policy: 100% success on 5/5 tasks, 100% safe success on 4/5 tasks; tortuous-tree sweep succeeded but breached the safety threshold.
- CathSim forward and sweep policies: 0% success on `phantom3_bca` and `phantom3_lcca` with 30 episodes each.
- CathSim pilot had no crashes and ran about 13.7-14.5 simulation steps/sec.
- Lumen pilot had no crashes and ran about 62-120 simulation steps/sec depending on task.
- stEVE/SOFA BasicWireNav pilot had no crashes. Forward and sweep each reached 20% success / 20% safe success and ran about 16-17 simulation steps/sec; random reached 0% success and ran about 28 simulation steps/sec.
- stEVE/SOFA ArchVariety pilot had no crashes. Forward reached 6.7% success / 6.7% safe success and sweep reached 13.3% success / 13.3% safe success; the task ran about 28 simulation steps/sec for guided policies and about 57 simulation steps/sec for random.
- A prior stEVE/SOFA all-task attempt including DualDeviceNav was stopped because the naive high-insertion pilot policy produced repeated `InterventionalRadiologyController` internal errors about `totalLengthIsChanging`. DualDeviceNav remains executable through the Docker stack, but it needs a reduced-action policy or the official trained-controller protocol before inclusion as benchmark evidence.

Short trained-agent result summary:

- Lumen `nav_tree_branch` PPO, 50,000 training steps, seed 0, 30 deterministic eval episodes: 100% success, 100% safe success, 0% crash rate, 0% unsafe event rate, mean 58.0 steps on successful episodes, 79.7 eval steps/sec. Result file: `results/lumen-tree-ppo-short-50k.json`.
- CathSim `phantom3_bca` PPO, 50,000 training steps, seed 0, 30 deterministic eval episodes: 100% success, 6.7% safe success, 0% crash rate, 93.3% unsafe event rate, mean 36.8 steps on successful episodes, 12.1 eval steps/sec. Result file: `results/cathsim-bca-ppo-short-50k.json`.

Decision status:

- The short PPO comparison is favorable to Lumen under the predeclared rule: Lumen is ahead on safe success, unsafe-event rate, crash-equivalent stability, and evaluation throughput; raw success is tied; CathSim has lower mean steps but the lower step count is paired with a high unsafe-event rate.
- The preprint was updated with the measured 50,000-step PPO comparison and the three-environment pilot summary.
- Full CathSim-style main results still require 600,000 training steps, six seeds, and 100 eval episodes per seed/task.
