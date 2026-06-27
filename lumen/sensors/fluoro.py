"""FluoroSensor (Layer 1): the device-in-lumen scene -> synthetic fluoroscopy.

The endovascular instance of the modality-agnostic sensor swap point (doc §3.9): it
turns a Layer-0 state (device node polyline; later the contrast lumen over R(s,θ)) into
a projective X-ray — the *native* clinical observation, not RGB (doc §3.6, §4). The
luminal-RGB sibling (`luminal.LuminalCamera`) sits beside it for bronchoscopy/GI.
"""

from __future__ import annotations

import numpy as np

from lumen.sensors.carm import CArm
from lumen.sensors.drr import radiograph, raycast
from lumen.sensors.realism import degrade
from lumen.sensors.volume import grid_for, voxelize_device


class FluoroSensor:
    """`res` is the μ-VOLUME resolution; `nu`/`nv` are the DETECTOR (image) pixels —
    independent knobs (raising one doesn't change the other)."""

    def __init__(self, mu_device=1.0, eps=0.6, res=64, n_samples=192, margin=8.0,
                 nu=128, nv=128):
        self.mu_device, self.eps = mu_device, eps
        self.res, self.n_samples, self.margin = res, n_samples, margin
        self.nu, self.nv = nu, nv

    def default_carm(self, nodes, axis=(1.0, 0.0, 0.0), **kw):
        """A C-arm centred on the device, viewing along `axis`, sized to cover the
        scene. `span` = device extent + 2·margin (matches grid_for's box). The factors
        place the source 2·span back, the detector 4·span across the scene (so the
        beam passes through it), and make the detector 1.6·span wide — a small FOV
        margin so the projected scene fits with room to spare."""
        c = np.asarray(nodes, float).mean(0)
        span = float(np.ptp(np.asarray(nodes, float), axis=0).max()) + 2 * self.margin
        return CArm.looking_at(c, distance=2.0 * span, axis=axis, sdd=4.0 * span,
                               width=1.6 * span, height=1.6 * span,
                               nu=self.nu, nv=self.nv, **kw)

    def render(self, nodes, radius=0.2, carm: CArm | None = None, beer_lambert=False,
               realism=None, contrast_nodes=None, contrast_radius=1.5, mu_contrast=0.25,
               contrast_eps=1.0):
        """Render the device polyline to a DRR line-integral image (or a Beer–Lambert
        radiograph if beer_lambert=True). Returns (image, carm).

        Pass `realism` (a RealismParams) to apply the detector-physics seam (L1.4 —
        Poisson noise, PSF, scatter, beam hardening) to the line integral before the
        optional Beer–Lambert step; default None leaves the clean DRR unchanged.

        Pass `contrast_nodes` to add a low-attenuation contrast-filled lumen roadmap
        behind the radio-opaque device. This makes synthetic fluoro usable for CV
        tasks that need vessel context instead of an isolated wire on a blank field."""
        nodes = np.asarray(nodes, float)
        scene_nodes = nodes if contrast_nodes is None else np.concatenate(
            [nodes, np.asarray(contrast_nodes, float)], axis=0)
        carm = carm or self.default_carm(scene_nodes)
        grid = grid_for(scene_nodes, margin=self.margin, res=self.res)
        mu = voxelize_device(nodes, radius, grid, mu_device=self.mu_device, eps=self.eps)
        if contrast_nodes is not None and mu_contrast:
            mu = mu + voxelize_device(contrast_nodes, contrast_radius, grid,
                                      mu_device=mu_contrast, eps=contrast_eps)
        A = raycast(mu, grid, carm, n_samples=self.n_samples)
        if realism is not None:
            A = degrade(A, realism)
        return (radiograph(A) if beer_lambert else A), carm
