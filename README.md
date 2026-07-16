# Lumen

**Open, differentiable, GPU-parallel simulation for wall-safe endovascular AI.**

Lumen is an Apache-2.0 research environment for training and evaluating agents that navigate slender devices through deformable vascular anatomy. It combines Newton/Warp-backed simulation, tube-intrinsic contact, synthetic fluoroscopy, luminal RGB, CV labels, replayable datasets, Gymnasium environments, and benchmark scoring that ranks safe target reach above raw reach.

Launch page, demo video, screenshots, and preprint:
https://seldingermed.github.io/seldinger-lumen/

![Lumen launch still](docs/assets/launch/social-card.png)

## Benchmark Snapshot

In a matched branch-navigation PPO comparison using 50,000 training steps and 30 deterministic evaluation episodes, Lumen reached 100% raw success and 100% safe success on `nav_tree_branch`. CathSim reached 100% raw success on `phantom3_bca`, but 6.7% safe success under the comparison force threshold. Lumen evaluation throughput was 79.7 steps/s versus 12.1 steps/s for CathSim in this run.

The full preprint and benchmark summaries are linked from the [launch page](https://seldingermed.github.io/seldinger-lumen/).

## Install

```bash
git clone https://github.com/SeldingerMed/seldinger-lumen
cd seldinger-lumen
pip install -e ".[dev]"
lumen doctor
```

## First Run

```bash
lumen play stenotic --out lumen-run
lumen benchmark lumen-bench
lumen render-fluoro lumen-fluoro.png
lumen capture lumen-episodes
lumen validate lumen-episodes --require-cv-labels
lumen index lumen-episodes --out lumen-episodes/index.jsonl --check-sidecars
lumen split-index lumen-episodes/index.jsonl --out-dir lumen-episodes/splits
```

## Python API

```python
import gymnasium as gym
import lumen.envs.registration

env = gym.make("Lumen/NavStenotic-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## What Is Included

- Procedural stenotic, tortuous, and branching vascular cases.
- Tube-intrinsic contact with wall penetration and safety status.
- Synthetic fluoroscopy, masks, keypoints, detector noise, and luminal RGB.
- Flow-diverter, aneurysm-inflow, clot, retrieval, and fragmentation modules.
- Dataset capture, validation, indexing, splitting, and materialization tooling.

Solver feature coverage is tracked in [docs/SOLVER_SUPPORT.md](docs/SOLVER_SUPPORT.md).

## Citation

```bibtex
@software{son_lumen_2026,
  author = {Son, Colin},
  title = {Lumen: an Open, Differentiable, GPU-Parallel Environment for Endovascular AI},
  year = {2026},
  url = {https://github.com/SeldingerMed/seldinger-lumen},
  license = {Apache-2.0}
}
```

## License

Apache-2.0.
