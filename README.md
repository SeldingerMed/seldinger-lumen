# Lumen

**Train endovascular navigation against the vessel wall, not just the target.**

Lumen is an Apache-2.0 simulation and RL environment for catheter and guidewire navigation. It combines deformable vascular anatomy, finite-radius contact, synthetic imaging, replayable episodes, and Gymnasium tasks in one GPU-parallel solver.

[Launch page](https://seldingermed.github.io/seldinger-lumen/) · [Preprint](https://seldingermed.github.io/seldinger-lumen/assets/launch/lumen-preprint.pdf) · [Solver coverage](docs/SOLVER_SUPPORT.md)

![Lumen advanced simulator captures](docs/assets/launch/physics-layer.png)

## Benchmark Snapshot

In a matched branch-navigation PPO comparison using 50,000 training steps and 30 deterministic evaluation episodes, Lumen reached 100% raw success and 100% safe success on `nav_tree_branch`. CathSim reached 100% raw success on `phantom3_bca`, but 6.7% safe success under the comparison force threshold. Lumen evaluation throughput was 79.7 steps/s versus 12.1 steps/s in this run.

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

## Core systems

- Procedural stenotic, tortuous, aneurysmal, and branching vessels.
- Fixed-port guidewire and coaxial catheter actuation with rotation, latency, backlash, and motion limits.
- Deformable wall mechanics, finite-radius contact, anisotropic friction, flow, clot, retrieval, and flow diversion.
- Synthetic fluoroscopy, luminal RGB, masks, keypoints, noise, latency, and dropout.
- Dataset capture, validation, replay, indexing, splitting, and materialization.
- Privileged, tracked, and raw-image observation contracts for RL.

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
