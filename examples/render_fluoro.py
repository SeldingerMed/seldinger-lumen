"""Render a guidewire to synthetic fluoroscopy (Layer 1 L1.0) and save previews.

    python examples/render_fluoro.py [out.png]

Pure numpy + stdlib PNG writer (no matplotlib/PIL). Uses a synthetic curved wire; to
render a live device, pass NewtonGuidewireSim.body_positions() as `nodes`.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from lumen.sensors import FluoroSensor, write_avi, write_png


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fluoro.png")
    a = np.linspace(0, np.pi / 2, 40)
    wire = np.stack([30 * np.sin(a) + 3 * np.sin(6 * a), 2 * np.cos(3 * a),
                     30 * (1 - np.cos(a))], axis=1)
    # render the DRR line integral A (device = high attenuation); min-max is the
    # standard DRR display (A has no fixed range) and shows the device BRIGHT. For the
    # clinical Beer-Lambert look (dark device on a bright field) use beer_lambert=True
    # and scale I*255 directly (I is already in [0,1]).
    vessel = np.stack([30 * np.sin(a), np.zeros_like(a),
                       30 * (1 - np.cos(a))], axis=1)
    sensor = FluoroSensor(mu_device=1.2, res=96, n_samples=260)
    views = sensor.render_biplanar(wire, radius=0.6, contrast_nodes=vessel,
                                   contrast_radius=2.0, mu_contrast=0.16)
    write_png(out, np.flipud(views[0]["image"]))
    stem = out.parent / out.stem
    write_png(stem.parent / f"{stem.name}_lateral.png", np.flipud(views[1]["image"]))
    write_png(stem.parent / f"{stem.name}_device_mask.png",
              np.flipud(views[0]["masks"]["device"].astype(float)))
    write_png(stem.parent / f"{stem.name}_vessel_mask.png",
              np.flipud(views[0]["masks"]["vessel"].astype(float)))
    write_avi(stem.parent / f"{stem.name}_biplanar.avi", [np.flipud(v["image"]) for v in views],
              fps=2)
    tip = views[0]["keypoints"]["tip"]["uv"]
    tip = (tip[0], views[0]["image"].shape[0] - 1 - tip[1])
    print(f"wrote {out}, {stem}_lateral.png, masks, and {stem}_biplanar.avi; "
          f"tip keypoint view0=({tip[0]:.1f}, {tip[1]:.1f})")


if __name__ == "__main__":
    main()
