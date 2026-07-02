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
from lumen.sensors.volume import asset_points, edge_radii, grid_for, voxelize_asset, voxelize_device


def _projected_radius_px(carm: CArm, point, radius):
    if radius <= 0:
        return 0.0
    uv0 = np.asarray(carm.project(point), float)
    if not np.isfinite(uv0).all():
        return 0.0
    u_axis, v_axis, _ = carm.axes()
    scales = []
    for axis in (u_axis, v_axis):
        uv = np.asarray(carm.project(np.asarray(point, float) + float(radius) * axis), float)
        if np.isfinite(uv).all():
            scales.append(float(np.linalg.norm(uv - uv0)))
    return max(scales) if scales else 0.0


def _project_polyline_mask(carm: CArm, nodes, shape, radius_px=2.0):
    nodes = np.asarray(nodes, float)
    h, w = shape
    uv = np.array([carm.project(p) for p in nodes], float)
    radii = np.asarray(radius_px, float)
    if radii.ndim == 0:
        radii = np.full(len(nodes), float(radii))
    if len(radii) != len(nodes):
        raise ValueError("radius_px must be scalar or one value per polyline node")
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = np.full((h, w), np.inf)
    if len(uv) == 1:
        u, v = uv[0]
        if np.isfinite([u, v]).all():
            d2 = (xx - u) ** 2 + (yy - v) ** 2
            return d2 <= max(float(radii[0]), 0.0) ** 2
    else:
        for i, (a, b) in enumerate(zip(uv[:-1], uv[1:])):
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
            r = max(float(radii[i]), float(radii[i + 1]), 0.0)
            d2 = np.minimum(d2, dist / max(r * r, 1e-12))
    return d2 <= 1.0


def _project_asset_mask(carm: CArm, asset, shape):
    mask = np.zeros(shape, dtype=bool)
    for edge in asset.edges:
        pts = np.asarray(edge.centerline_mm, float)
        radii = edge_radii(edge, pts)
        px = [max(1.0, _projected_radius_px(carm, p, r)) for p, r in zip(pts, radii)]
        mask |= _project_polyline_mask(carm, pts, shape, px)
    return mask


def _keypoint(carm: CArm, point, shape):
    u, v = carm.project(point)
    present = bool(np.isfinite([u, v]).all() and 0.0 <= u < shape[1] and 0.0 <= v < shape[0])
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
               contrast_eps=1.0, contrast_asset=None):
        """Render the device polyline to a DRR line-integral image (or a Beer–Lambert
        radiograph if beer_lambert=True). Returns (image, carm).

        Pass `realism` (a RealismParams) to apply the detector-physics seam (L1.4 —
        Poisson noise, PSF, scatter, beam hardening) to the line integral before the
        optional Beer–Lambert step; default None leaves the clean DRR unchanged.

        Pass `contrast_nodes` to add a low-attenuation contrast-filled lumen roadmap
        behind the radio-opaque device, or pass `contrast_asset` to rasterize every
        edge of a Lumen asset with its stored radius profile."""
        nodes = np.asarray(nodes, float)
        has_asset = contrast_asset is not None and bool(mu_contrast)
        has_contrast = contrast_nodes is not None and bool(mu_contrast)
        if has_asset:
            contrast_pts = asset_points(contrast_asset)
        elif has_contrast:
            contrast_pts = np.asarray(contrast_nodes, float)
        else:
            contrast_pts = None
        scene_nodes = nodes if contrast_pts is None else np.concatenate([nodes, contrast_pts], axis=0)
        carm = carm or self.default_carm(scene_nodes)
        grid = grid_for(scene_nodes, margin=self.margin, res=self.res)
        mu = voxelize_device(nodes, radius, grid, mu_device=self.mu_device, eps=self.eps)
        if has_asset:
            mu = mu + voxelize_asset(contrast_asset, grid, mu_device=mu_contrast,
                                     eps=contrast_eps)
        if has_contrast:
            mu = mu + voxelize_device(contrast_nodes, contrast_radius, grid,
                                      mu_device=mu_contrast, eps=contrast_eps)
        A = raycast(mu, grid, carm, n_samples=self.n_samples)
        if realism is not None:
            A = degrade(A, realism)
        return (radiograph(A) if beer_lambert else A), carm

    def render_scene(self, nodes, radius=0.2, carm: CArm | None = None,
                     beer_lambert=False, realism=None, contrast_nodes=None,
                     contrast_radius=1.5, mu_contrast=0.25, contrast_eps=1.0,
                     contrast_asset=None):
        """Render image plus CV supervision products: masks and keypoints."""
        img, carm = self.render(nodes, radius=radius, carm=carm, beer_lambert=beer_lambert,
                                realism=realism, contrast_nodes=contrast_nodes,
                                contrast_radius=contrast_radius, mu_contrast=mu_contrast,
                                contrast_eps=contrast_eps, contrast_asset=contrast_asset)
        nodes = np.asarray(nodes, float)
        device_px = [max(1.0, _projected_radius_px(carm, p, radius)) for p in nodes]
        masks = {"device": _project_polyline_mask(carm, nodes, img.shape, device_px)}
        if contrast_asset is not None:
            masks["vessel"] = _project_asset_mask(carm, contrast_asset, img.shape)
        elif contrast_nodes is not None:
            contrast_nodes = np.asarray(contrast_nodes, float)
            vessel_px = [max(1.0, _projected_radius_px(carm, p, contrast_radius))
                         for p in contrast_nodes]
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
        contrast_asset = kw.get("contrast_asset")
        has_contrast = bool(kw.get("mu_contrast", 0.25))
        if contrast_asset is not None and has_contrast:
            contrast_pts = asset_points(contrast_asset)
        elif contrast_nodes is not None and has_contrast:
            contrast_pts = np.asarray(contrast_nodes, float)
        else:
            contrast_pts = None
        scene_nodes = nodes if contrast_pts is None else np.concatenate([nodes, contrast_pts], axis=0)
        if carms is None:
            carms = [self.default_carm(scene_nodes, axis=axis) for axis in axes]
        return [self.render_scene(nodes, carm=c, **kw) for c in carms]
