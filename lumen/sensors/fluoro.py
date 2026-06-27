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


def _project_polyline_mask(carm: CArm, nodes, shape, radius_px=2.0):
    nodes = np.asarray(nodes, float)
    h, w = shape
    uv = np.array([carm.project(p) for p in nodes], float)
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = np.full((h, w), np.inf)
    if len(uv) == 1:
        u, v = uv[0]
        if np.isfinite([u, v]).all():
            d2 = (xx - u) ** 2 + (yy - v) ** 2
    else:
        for a, b in zip(uv[:-1], uv[1:]):
            if not np.isfinite([*a, *b]).all():
                continue
            ab = b - a
            den = float(ab @ ab)
            if den < 1e-12:
                dist = (xx - a[0]) ** 2 + (yy - a[1]) ** 2
            else:
                t = np.clip(((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / den, 0.0, 1.0)
                px = a[0] + t * ab[0]
                py = a[1] + t * ab[1]
                dist = (xx - px) ** 2 + (yy - py) ** 2
            d2 = np.minimum(d2, dist)
    return d2 <= radius_px ** 2


def _keypoint(carm: CArm, point, shape):
    u, v = carm.project(point)
    present = bool(np.isfinite([u, v]).all() and -0.5 <= u < shape[1] - 0.5
                   and -0.5 <= v < shape[0] - 0.5)
    return {"uv": (float(u), float(v)), "present": present}


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

    def render_scene(self, nodes, radius=0.2, carm: CArm | None = None,
                     beer_lambert=False, realism=None, contrast_nodes=None,
                     contrast_radius=1.5, mu_contrast=0.25, contrast_eps=1.0):
        """Render image plus CV supervision products: masks and keypoints."""
        img, carm = self.render(nodes, radius=radius, carm=carm, beer_lambert=beer_lambert,
                                realism=realism, contrast_nodes=contrast_nodes,
                                contrast_radius=contrast_radius, mu_contrast=mu_contrast,
                                contrast_eps=contrast_eps)
        nodes = np.asarray(nodes, float)
        device_px = max(1.0, radius / max(carm.width / carm.nu, 1e-9))
        masks = {"device": _project_polyline_mask(carm, nodes, img.shape, device_px)}
        if contrast_nodes is not None:
            vessel_px = max(1.0, contrast_radius / max(carm.width / carm.nu, 1e-9))
            masks["vessel"] = _project_polyline_mask(carm, contrast_nodes, img.shape, vessel_px)
        else:
            masks["vessel"] = np.zeros_like(masks["device"], dtype=bool)
        kpts = {"base": _keypoint(carm, nodes[0], img.shape),
                "tip": _keypoint(carm, nodes[-1], img.shape),
                "nodes": [_keypoint(carm, p, img.shape) for p in nodes]}
        return {"image": img, "carm": carm, "masks": masks, "keypoints": kpts}

    def render_biplanar(self, nodes, axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), carms=None,
                        **kw):
        """Render two calibrated fluoro views of the same scene."""
        nodes = np.asarray(nodes, float)
        contrast_nodes = kw.get("contrast_nodes")
        scene_nodes = nodes if contrast_nodes is None else np.concatenate(
            [nodes, np.asarray(contrast_nodes, float)], axis=0)
        if carms is None:
            carms = [self.default_carm(scene_nodes, axis=axis) for axis in axes]
        return [self.render_scene(nodes, carm=c, **kw) for c in carms]
