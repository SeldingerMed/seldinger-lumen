"""L2.1 — capture procedural episodes into the Layer-2 schema.

    python examples/capture_episode.py [out_dir]

Generates a few procedural cases (straight / stenotic), runs the guidewire to the
target while recording the paired observation each step, and writes one replayable
case-bundle directory per case under <out_dir>. Reloads them and prints a summary.
Needs the full stack (newton + warp).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from lumen.assets import procedural
from lumen.data import CaseBundle, Episode, rollout_episode, validate
from lumen.sensors import FluoroSensor, LuminalCamera, write_png


def _write_preview_sheet(ep, root: Path) -> tuple[Path, Path, Path | None]:
    obs_steps = [s for s in ep.steps if s.obs_ref]
    if not obs_steps:
        raise ValueError("episode has no observation sidecars to preview")
    picks = [0, len(obs_steps) // 2, len(obs_steps) - 1]
    frames = [obs_steps[i].load_obs(root) for i in picks]
    preview = root / "preview.png"
    sheet = root / "preview_contact_sheet.png"
    write_png(preview, frames[0])
    write_png(sheet, np.concatenate(frames, axis=1))
    masks = [obs_steps[i].load_annotation(root, "device_mask") for i in picks]
    mask_sheet = None
    if all(m is not None for m in masks):
        mask_sheet = root / "device_mask_contact_sheet.png"
        write_png(mask_sheet, np.concatenate([m.astype(float) for m in masks], axis=1))
    vessel_masks = [obs_steps[i].load_annotation(root, "vessel_mask") for i in picks]
    if all(m is not None for m in vessel_masks):
        write_png(root / "vessel_mask_contact_sheet.png",
                  np.concatenate([m.astype(float) for m in vessel_masks], axis=1))
    return preview, sheet, mask_sheet


def main(out_dir="episodes"):
    out_root = Path(out_dir)
    sensor = FluoroSensor(res=32, nu=64, nv=64, n_samples=96)
    cases = {
        "straight_fluoro": dict(asset=procedural.straight_tube(80.0, 2.0), sensor=sensor),
        "stenosis_fluoro": dict(asset=procedural.stenotic_tube(80.0, 2.0, severity=0.6), sensor=sensor),
        "straight_luminal": dict(asset=procedural.straight_tube(80.0, 4.0),
                                 sensor=LuminalCamera(nu=64, nv=64), modality="luminal"),
    }
    for name, kw in cases.items():
        path = out_root / name
        print(f"capturing {name:18s} -> {path}", flush=True)
        ep = rollout_episode(max_steps=30, asset_ref=f"{name}.asset.json", label=name,
                             notes={"case": name, "true_C10": 4000.0}, **kw)
        validate(ep)
        ep.save(path)
        back = Episode.load(path)
        validate(back, root=path)
        bundle = CaseBundle.load(path)
        obs0 = back.steps[0].load_obs(path)
        preview, sheet, mask_sheet = _write_preview_sheet(back, path)
        tip_ok = back.outcome.metrics["tip_target"]["success"]
        wall_risk = back.outcome.metrics["wall_safety"]["perforation_risk"]
        mask_msg = f"  mask_sheet={mask_sheet}" if mask_sheet else ""
        vessel_sheet = path / "vessel_mask_contact_sheet.png"
        vessel_msg = f"  vessel_sheet={vessel_sheet}" if vessel_sheet.exists() else ""
        print(f"{name:18s}  steps={back.outcome.steps:2d}  success={back.outcome.success!s:5s}  "
              f"final_dist={back.outcome.final_dist:6.2f}  obs{obs0.shape}  "
              f"calib={bundle.calibration['type']}  tip_target={tip_ok!s:5s}  "
              f"wall_risk={wall_risk!s:5s}  preview={preview}  sheet={sheet}"
              f"{mask_msg}{vessel_msg}",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "episodes")
