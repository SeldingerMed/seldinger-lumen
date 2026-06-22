"""FluoroSensor (Layer 1): the device-in-lumen scene -> synthetic fluoroscopy.

The endovascular instance of the modality-agnostic sensor swap point (doc §3.9): it
turns a Layer-0 state (device node polyline; later the contrast lumen over R(s,θ)) into
a projective X-ray — the *native* clinical observation, not RGB (doc §3.6, §4). A
luminal-RGB sibling sensor will sit beside it for bronchoscopy/GI.
"""

from __future__ import annotations

import numpy as np

from lumen.sensors.carm import CArm
from lumen.sensors.drr import radiograph, raycast
from lumen.sensors.volume import grid_for, voxelize_device


class FluoroSensor:
    def __init__(self, mu_device=1.0, eps=0.6, res=64, n_samples=192, margin=8.0):
        self.mu_device, self.eps = mu_device, eps
        self.res, self.n_samples, self.margin = res, n_samples, margin

    def default_carm(self, nodes, axis=(1.0, 0.0, 0.0), **kw):
        """A C-arm centred on the device, viewing along `axis`."""
        c = np.asarray(nodes, float).mean(0)
        span = float(np.ptp(np.asarray(nodes, float), axis=0).max()) + 2 * self.margin
        return CArm.looking_at(c, distance=2.0 * span, axis=axis, sdd=4.0 * span,
                               width=1.6 * span, height=1.6 * span, **kw)

    def render(self, nodes, radius=0.2, carm: CArm | None = None, beer_lambert=False):
        """Render the device polyline to a DRR line-integral image (or a Beer–Lambert
        radiograph if beer_lambert=True). Returns (image, carm)."""
        nodes = np.asarray(nodes, float)
        carm = carm or self.default_carm(nodes)
        grid = grid_for(nodes, margin=self.margin, res=self.res)
        mu = voxelize_device(nodes, radius, grid, mu_device=self.mu_device, eps=self.eps)
        A = raycast(mu, grid, carm, n_samples=self.n_samples)
        return (radiograph(A) if beer_lambert else A), carm
