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
> A historical 50k-step PPO run measured 100% raw target reach in both native tasks and 6.6× higher Lumen evaluation throughput (79.7 vs 12.1 steps/s). The safety endpoints are not commensurate: Lumen's archived result used centerline penetration in simulator units; CathSim reports native contact force. The current contract publishes those traces separately and makes no cross-environment safety claim.
>
> The simulator emits route progress, native wall-overlap/load curves, fluoroscopy, masks, keypoints, and replay metadata from the same scene.
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

> I launched Lumen, an Apache-2.0 simulator for endovascular RL. It reports native surface-overlap and wall-load traces alongside raw target reach, with procedural vascular cases, tube-intrinsic contact, synthetic fluoro, masks/keypoints, luminal RGB, replayable datasets, and Gymnasium benchmarks.
>
> Repo: https://github.com/SeldingerMed/seldinger-lumen
> Launch page/video/preprint: https://seldingermed.github.io/seldinger-lumen/

Technical:

> The current release includes procedural stenotic/tortuous/branching vessels, native surface-overlap and wall-load curves in simulator units, synthetic fluoro with CV labels, luminal RGB, replayable episode capture, and reduced-order modules for aneurysm inflow, flow diversion, clot, stentriever retrieval, and fragmentation.
>
> Native safety endpoints are deliberately not collapsed across simulators. Physical force calibration requires matched device geometry, anatomy/materials, solver-unit verification, and phantom or ex-vivo force--injury validation.

## Reddit

Title options:

- Lumen: open-source wall-safe endovascular RL environment
- Launching Lumen, an Apache-2.0 simulator for endovascular RL
- Open simulator for endovascular navigation with safety scoring, fluoro, and CV labels

Body:

> I just launched Lumen, an Apache-2.0 simulator for endovascular RL research.
>
> The goal is to make endovascular navigation trainable as an auditable benchmark rather than a simple target-reaching task. Lumen includes procedural vascular cases, tube-intrinsic contact, native wall-overlap/load curves in simulator units, synthetic fluoroscopy, masks/keypoints, luminal RGB, dataset capture/validation/indexing, and Gymnasium environments.
>
> It also includes advanced state modules for aneurysm inflow, flow diversion, clot fields, stentriever retrieval, and fragmentation.
>
> A historical matched branch-navigation PPO run recorded 100% raw success on each native task. The safety traces remain endpoint-labelled and are not a cross-environment safety comparison.
>
> Repo: https://github.com/SeldingerMed/seldinger-lumen
> Launch page/video/preprint: https://seldingermed.github.io/seldinger-lumen/
>
> I would be especially interested in feedback from people working on robotic endovascular navigation, medical simulation, synthetic fluoro/CV data, or RL benchmark design.

## Reply Starters

For "How is this different from CathSim?":

> CathSim is the key prior open simulator and helped make autonomous catheterization easier to study. Lumen focuses on wall-overlap/load observability, deformable-wall semantics, paired state/image observations, synthetic CV labels, replayable dataset tooling, and modules for aneurysm/flow/clot/device state.
>
> The environments' native safety traces are reported separately. No cross-environment safety rate is claimed until matched physical calibration exists.

For "Is this clinically validated?":

> The launch release is a research environment. The immediate value is reproducible experimentation around wall-safe navigation, imaging observations, labels, and endovascular state modules in a public Apache-2.0 repo.

For "Can I train agents on it?":

> Yes. The environments are Gymnasium-compatible, and the CLI includes benchmark/capture/validate/index/split tooling so policies and datasets can be replayed and compared.
