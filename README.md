# Lumen

**Train endovascular navigation against the vessel wall, not just the target.**

Lumen is an Apache-2.0 simulation and RL environment for catheter and guidewire navigation. It combines deformable vascular anatomy, finite-radius contact, synthetic imaging, replayable episodes, and Gymnasium tasks in one GPU-parallel solver.

[Launch page](https://seldingermed.github.io/seldinger-lumen/) · [Preprint](https://seldingermed.github.io/seldinger-lumen/assets/launch/lumen-preprint.pdf) · [Solver coverage](docs/SOLVER_SUPPORT.md)

![Lumen advanced simulator captures](docs/assets/launch/physics-layer.png)

## Benchmark Snapshot

The historical 50,000-step branch-navigation pilot measured 100% raw target reach in
both native environments and 79.7 versus 12.1 evaluation steps/s for Lumen versus
CathSim. Its safety fields are **not cross-environment comparable**: Lumen recorded
centerline penetration in simulator units while CathSim recorded native contact force.
The current benchmark contract reports those endpoints separately; no cross-environment
safety claim is made until matched device/anatomy/material calibration and
force–injury validation exist. The frozen pilot artifacts remain available for
provenance and must not be reinterpreted under the new contract.

The full preprint and benchmark summaries are linked from the [launch page](https://seldingermed.github.io/seldinger-lumen/).

The scaled protocol is an explicit disjoint-case check:

```bash
lumen benchmark bench_results --suite scaled --episodes 100
```

It evaluates the same policy on frozen procedural train and held-out case IDs,
reports raw/native-safe/crash rates for each split, and records the train-minus-held-out
generalization gap in `forward-baseline-heldout.json`.

Canonical scorecards also include episode-level `lumen-stats/1` summaries: fractional
interquartile means, ordinary means, and deterministic 95% percentile-bootstrap
intervals with recorded seeds and resample counts.

Generated scorecards include a SHA-256 certificate for every action sequence and native
episode outcome. Re-run the canonical baseline certificate locally with:

```bash
python tools/check_replay.py
```

Use `lumen.bench.replay_verified_leaderboard(results_dir, policies)` to rank only
scorecards whose certificates re-run successfully with the supplied policy callables.

## Install

```bash
git clone https://github.com/SeldingerMed/seldinger-lumen
cd seldinger-lumen
pip install -e ".[dev]"
lumen doctor
```

## CPU Docker image

The repository ships a CPU-only runtime image. It installs the pinned Newton/Warp
solver but does not require a CUDA device:

```bash
docker build --file docker/Dockerfile --tag seldinger-lumen:0.2.0 .
docker run --rm seldinger-lumen:0.2.0 anatomy --validate
```

## First Run

```bash
lumen play stenotic --out lumen-run
lumen benchmark lumen-bench
lumen anatomy --validate
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
- Simulator/phantom/device deployment interface with fail-closed safety envelopes and
  force/torque benchtop validation ([protocol](docs/DEPLOYMENT.md)).

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
