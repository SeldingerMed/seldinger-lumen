# Lumen Launch Social Drafts

Use the real screenshots and video from `docs/assets/launch/`. Recommended attachments:

- X/LinkedIn: `social-card.png` or `lumen-launch.mp4`
- Reddit/Discord: `sensor-layer.png`, `physics-layer.png`, and `lumen-launch.mp4`
- Square preview: `social-card-square.png`

## X

Post 1:

> Launching Lumen: an open, Apache-2.0 environment for wall-safe endovascular AI.
>
> It brings deformable vascular anatomy, tube-intrinsic contact, synthetic fluoroscopy, luminal RGB, masks/keypoints, replayable datasets, and Gymnasium RL benchmarks into one public stack.
>
> Repo: https://github.com/SeldingerMed/seldinger-lumen
> Page: https://seldingermed.github.io/seldinger-lumen/

Post 2:

> The benchmark target is not just "reach the branch."
>
> In Lumen, safe target reach is distinct from target reach with unsafe wall interaction. The simulator emits route progress, wall penetration, safety status, fluoroscopy, masks, keypoints, and replay metadata from the same scene.
>
> https://github.com/SeldingerMed/seldinger-lumen

Post 3:

> Lumen goes beyond rigid-pipe catheter tasks:
>
> - deformable-wall semantics
> - tube-intrinsic contact
> - synthetic biplanar fluoro
> - luminal RGB
> - dataset capture/validation/indexing
> - clot, aneurysm, flow-diverter, retrieval state
>
> Open repo + preprint:
> https://seldingermed.github.io/seldinger-lumen/

## Discord

Short:

> I launched Lumen, an open Apache-2.0 simulator for endovascular AI. It is built around wall-safe navigation rather than simple target reach: deformable vascular cases, tube-intrinsic contact, synthetic fluoro, masks/keypoints, luminal RGB, replayable datasets, and Gymnasium benchmarks.
>
> Repo: https://github.com/SeldingerMed/seldinger-lumen
> Launch page/video/preprint: https://seldingermed.github.io/seldinger-lumen/

Technical:

> Lumen is meant to be a stronger open benchmark substrate for autonomous endovascular navigation. The current release includes procedural stenotic/tortuous/branching vessels, wall-penetration and safe-success metrics, synthetic fluoro with CV labels, luminal RGB, replayable episode capture, and reduced-order modules for aneurysm inflow, flow diversion, clot, stentriever retrieval, and fragmentation.
>
> The important distinction: a policy that reaches the target after unsafe wall interaction is not scored as the same thing as safe target reach.

## Reddit

Title options:

- Lumen: open-source wall-safe endovascular RL environment
- Launching Lumen, an Apache-2.0 simulator for endovascular AI
- Open simulator for endovascular navigation with safety scoring, fluoro, and CV labels

Body:

> I just launched Lumen, an Apache-2.0 environment for endovascular AI research.
>
> The goal is to make endovascular navigation trainable as a safety-scored benchmark rather than a simple target-reaching task. Lumen includes procedural vascular cases, tube-intrinsic contact, wall-penetration metrics, safe-success scoring, synthetic fluoroscopy, masks/keypoints, luminal RGB, dataset capture/validation/indexing, and Gymnasium environments.
>
> It also includes advanced state modules for aneurysm inflow, flow diversion, clot fields, stentriever retrieval, and fragmentation.
>
> Repo: https://github.com/SeldingerMed/seldinger-lumen
> Launch page/video/preprint: https://seldingermed.github.io/seldinger-lumen/
>
> I would be especially interested in feedback from people working on robotic endovascular navigation, medical simulation, synthetic fluoro/CV data, or RL benchmark design.

## Reply Starters

For "How is this different from CathSim?":

> CathSim is the key prior open simulator and helped make autonomous catheterization easier to study. Lumen is aimed at the next benchmark layer: wall-safety scoring, deformable-wall semantics, paired state/image observations, synthetic CV labels, replayable dataset tooling, and modules for aneurysm/flow/clot/device state.

For "Is this clinically validated?":

> The launch release is a research environment and benchmark substrate. The immediate value is reproducible experimentation around wall-safe navigation, imaging observations, labels, and endovascular state modules in a public Apache-2.0 stack.

For "Can I train agents on it?":

> Yes. The environments are Gymnasium-compatible, and the CLI includes benchmark/capture/validate/index/split tooling so policies and datasets can be replayed and compared.
