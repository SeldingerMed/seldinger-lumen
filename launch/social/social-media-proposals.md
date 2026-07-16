# Lumen Launch Social Drafts

Use the real screenshots and video from `docs/assets/launch/`. Recommended attachments:

- X/LinkedIn: `social-card.png` or `lumen-launch.mp4`
- Reddit/Discord: `sensor-layer.png`, `physics-layer.png`, and `lumen-launch.mp4`
- Square preview: `social-card-square.png`

## X

Post 1:

> Launching Lumen: an Apache-2.0 simulator for wall-safe endovascular RL.
>
> It includes procedural vascular cases, tube-intrinsic contact, synthetic fluoroscopy, luminal RGB, masks/keypoints, replayable datasets, and Gymnasium benchmarks.
>
> Repo: https://github.com/SeldingerMed/seldinger-lumen
> Page: https://seldingermed.github.io/seldinger-lumen/

Post 2:

> The benchmark target is not just "reach the branch."
>
> In a matched 50k-step PPO branch-navigation comparison, Lumen and CathSim both reached 100% raw success, but Lumen preserved 100% safe success vs 6.7% for CathSim under the force threshold.
>
> The simulator emits route progress, wall penetration, safety status, fluoroscopy, masks, keypoints, and replay metadata from the same scene.
>
> https://github.com/SeldingerMed/seldinger-lumen

Post 3:

> What is in the current release:
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

> I launched Lumen, an Apache-2.0 simulator for endovascular RL. It scores wall-safe navigation separately from raw target reach and includes procedural vascular cases, tube-intrinsic contact, synthetic fluoro, masks/keypoints, luminal RGB, replayable datasets, and Gymnasium benchmarks.
>
> Repo: https://github.com/SeldingerMed/seldinger-lumen
> Launch page/video/preprint: https://seldingermed.github.io/seldinger-lumen/

Technical:

> The current release includes procedural stenotic/tortuous/branching vessels, wall-penetration and safe-success metrics, synthetic fluoro with CV labels, luminal RGB, replayable episode capture, and reduced-order modules for aneurysm inflow, flow diversion, clot, stentriever retrieval, and fragmentation.
>
> The important distinction: a policy that reaches the target after unsafe wall interaction is not scored as the same thing as safe target reach. In the launch comparison, Lumen hit 100% raw success and 100% safe success after 50k PPO steps on branch navigation; CathSim hit 100% raw success but 6.7% safe success under the force threshold.

## Reddit

Title options:

- Lumen: open-source wall-safe endovascular RL environment
- Launching Lumen, an Apache-2.0 simulator for endovascular RL
- Open simulator for endovascular navigation with safety scoring, fluoro, and CV labels

Body:

> I just launched Lumen, an Apache-2.0 simulator for endovascular RL research.
>
> The goal is to make endovascular navigation trainable as a safety-scored benchmark rather than a simple target-reaching task. Lumen includes procedural vascular cases, tube-intrinsic contact, wall-penetration metrics, safe-success scoring, synthetic fluoroscopy, masks/keypoints, luminal RGB, dataset capture/validation/indexing, and Gymnasium environments.
>
> It also includes advanced state modules for aneurysm inflow, flow diversion, clot fields, stentriever retrieval, and fragmentation.
>
> In a matched branch-navigation PPO run, Lumen reached 100% raw success and 100% safe success over 30 held-out eval episodes. CathSim reached 100% raw success, but 6.7% safe success under the comparison force threshold.
>
> Repo: https://github.com/SeldingerMed/seldinger-lumen
> Launch page/video/preprint: https://seldingermed.github.io/seldinger-lumen/
>
> I would be especially interested in feedback from people working on robotic endovascular navigation, medical simulation, synthetic fluoro/CV data, or RL benchmark design.

## Reply Starters

For "How is this different from CathSim?":

> CathSim is the key prior open simulator and helped make autonomous catheterization easier to study. Lumen focuses on wall-safety scoring, deformable-wall semantics, paired state/image observations, synthetic CV labels, replayable dataset tooling, and modules for aneurysm/flow/clot/device state.
>
> In the launch PPO comparison, raw target reach tied at 100%, but Lumen preserved 100% safe success while CathSim fell to 6.7% safe success under the force threshold.

For "Is this clinically validated?":

> The launch release is a research environment. The immediate value is reproducible experimentation around wall-safe navigation, imaging observations, labels, and endovascular state modules in a public Apache-2.0 repo.

For "Can I train agents on it?":

> Yes. The environments are Gymnasium-compatible, and the CLI includes benchmark/capture/validate/index/split tooling so policies and datasets can be replayed and compared.
