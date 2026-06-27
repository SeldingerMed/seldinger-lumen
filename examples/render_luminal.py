"""L1.4 — render the second sensor modality (luminal RGB) and the fluoro realism seam.

    python examples/render_luminal.py [out_prefix]

Writes <prefix>_luminal.png (endoscopic RGB tunnel view from the device tip),
<prefix>_fluoro_noisy.png (the same scene as a realistic, dose-limited DRR), and
<prefix>_preview.avi. Pure numpy + stdlib exporters — no matplotlib/PIL.
"""

from __future__ import annotations

import sys

import numpy as np

from lumen.assets import procedural
from lumen.core.frame import CenterlineFrame
from lumen.sensors import FluoroSensor, LuminalCamera, RealismParams, write_avi, write_png


def main(prefix="out"):
    asset = procedural.stenotic_tube(80.0, 4.0, severity=0.6)   # a narrowing to see ahead
    pts, lumen = asset.edge_arrays(asset.edges[0])
    pts = np.asarray(pts)
    frame = CenterlineFrame(pts)
    device = np.stack([pts[1], pts[4]])                          # short device at the inlet, forward

    rgb = LuminalCamera(nu=128, nv=128, texture_strength=0.18, fold_strength=0.12,
                        artifact_strength=0.25, artifact_seed=0).render(frame, lumen, device)
    write_png(f"{prefix}_luminal.png", rgb)

    sensor = FluoroSensor(res=48, nu=128, nv=128)
    realism = RealismParams(i0=3e3, psf_sigma=1.0, scatter_frac=0.15, beam_hardening=0.05, seed=0)
    A, _ = sensor.render(device, contrast_nodes=pts, contrast_radius=3.5,
                         mu_contrast=0.18, realism=realism)      # realistic dose-limited DRR
    write_png(f"{prefix}_fluoro_noisy.png", A)
    write_avi(f"{prefix}_preview.avi", [rgb, np.repeat((A / (A.max() + 1e-9))[:, :, None], 3, axis=2)], fps=2)
    print(f"wrote {prefix}_luminal.png, {prefix}_fluoro_noisy.png, and {prefix}_preview.avi")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
