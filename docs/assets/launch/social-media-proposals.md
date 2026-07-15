# Lumen Launch Social Copy

Primary URL: https://seldingermed.github.io/seldinger-lumen/

Repository: https://github.com/SeldingerMed/seldinger-lumen

Suggested media:
- `social-card.png` for X/LinkedIn/Reddit link cards
- `social-card-square.png` for Discord/LinkedIn image posts
- `lumen-launch.mp4` for launch posts and pinned replies
- `lumen-preprint.pdf` for research communities

## X / Twitter

### Main Launch Post

I’m launching Lumen: an open, differentiable, GPU-parallel simulator for endovascular AI.

It gives RL/CV researchers:
- deformable vessel-wall physics
- Newton/Warp simulation
- synthetic fluoroscopy + masks/keypoints
- Gymnasium envs
- safety-scored benchmarks
- replayable dataset tooling

Repo + demo + preprint:
https://seldingermed.github.io/seldinger-lumen/

### Thread

1. Lumen is live.

Open, Apache-2.0 simulation for endovascular AI: guidewires, deformable lumens, synthetic fluoro, RL benchmarks, and CV labels in one stack.

https://seldingermed.github.io/seldinger-lumen/

2. The key idea: reaching the target is not enough.

Endovascular agents should not win by scraping through the vessel wall. Lumen reports safe success, unsafe success, wall penetration, and return, then ranks safe success first.

3. It is built around a deformable lumen field, not just a rigid pipe.

The wall is part of the simulation state. Contact, wall displacement, and safety metrics all refer to the same physical geometry.

4. It is also a CV data generator.

Lumen renders synthetic fluoroscopy with device masks, vessel masks, tip/base keypoints, node positions, and replayable case bundles.

5. The workflow is meant to be boring in a good way:

`lumen play`
`lumen benchmark`
`lumen capture`
`lumen validate`
`lumen index`
`lumen split-index`
`lumen materialize-batch`

6. CathSim helped make open endovascular RL concrete. Lumen pushes the open surface toward deformable-wall physics, differentiability, safety scoring, and dataset-grade CV artifacts.

7. Preprint, demo video, and repo:
https://seldingermed.github.io/seldinger-lumen/

### Short Follow-Up Posts

Lumen’s benchmark ranks safe target reach above raw target reach.

That sounds obvious, but it changes the incentives: an agent that reaches the branch by violating wall safety does not get treated as a clean success.

Lumen is open-source endovascular simulation for people who need more than a Gym wrapper:

deformable wall, contact mechanics, synthetic fluoro, masks/keypoints, replayable episodes, and safety-scored RL.

## Discord

I just launched Lumen, an open Apache-2.0 simulator for endovascular AI:
https://seldingermed.github.io/seldinger-lumen/

It is built for RL/CV work around guidewire/catheter navigation:
- differentiable Newton/Warp physics
- deformable vessel wall
- tube-intrinsic contact
- synthetic fluoroscopy
- masks/keypoints/node labels
- Gymnasium envs
- safety-scored benchmark
- replayable dataset tooling

The main difference from rigid-pipe catheter tasks is that Lumen makes wall safety part of the score, not a side note. A target reach that crosses the wall-safety threshold is reported as unsafe success.

Preprint and demo video are on the page.

## Reddit

Suggested title:

Open-source differentiable simulator for endovascular AI: deformable vessel wall, synthetic fluoro, safety-scored RL

Suggested body:

I’m releasing Lumen, an Apache-2.0 simulator for endovascular AI research:
https://github.com/SeldingerMed/seldinger-lumen

Launch page + preprint + demo:
https://seldingermed.github.io/seldinger-lumen/

The goal is to give RL/CV researchers a stronger open substrate for guidewire/catheter navigation than target-reaching in rigid tubes.

What is included:
- differentiable Newton/Warp physics
- deformable HGO-style vessel wall
- tube-intrinsic contact
- synthetic fluoroscopy
- masks, keypoints, device nodes, and replayable case bundles
- Gymnasium environments
- benchmark scoring that separates safe success from unsafe success
- dataset tooling: capture, validate, index, split, materialize

The benchmark ranks safe success before raw target reach, so an agent cannot “win” by reaching the target through unsafe wall interaction.

I’d be especially interested in feedback from people working on surgical robotics, catheter navigation, sim-to-real, or medical CV datasets.

## LinkedIn

I’m launching Lumen: an open, differentiable, GPU-parallel simulator for endovascular AI.

The research problem is not just “can an agent reach a point in a vessel?” It is whether it can navigate a slender device through deformable anatomy while preserving wall safety and producing image-grounded behavior that can be studied, replayed, and trained on.

Lumen combines:
- deformable vessel-wall physics
- Newton/Warp simulation
- tube-intrinsic contact
- synthetic fluoroscopy
- masks, keypoints, and node labels
- Gymnasium environments
- safety-scored RL benchmarks
- replayable dataset tooling

The launch package includes a preprint, demo video, and public Apache-2.0 repo:
https://seldingermed.github.io/seldinger-lumen/

## Suggested Replies

### “How is this different from CathSim?”

CathSim is the strongest open reference point for endovascular RL. Lumen targets a different surface: deformable-lumen physics, differentiability, safety-scored success, synthetic fluoro labels, and replayable dataset tooling in an Apache-2.0 stack.

### “Is this only for vascular work?”

The launch focus is endovascular because that is where the benchmark and fluoro tooling are aimed. The core abstraction is broader: a slender device moving through a deformable lumen, which also maps to airway, bowel, duct, and scope-like tasks.

### “Can it generate training data?”

Yes. Lumen can capture replayable case bundles, validate labels/sidecars, write JSONL dataloader indexes, split by episode to avoid leakage, and materialize strict NPZ smoke-test batches.

### “What should I try first?”

Start with:

```bash
pip install -e ".[dev]"
lumen doctor
lumen play stenotic --out lumen-run
lumen benchmark lumen-bench
```

Then try the capture/validate/index workflow if you care about CV or imitation-learning data.
