"""Luminal RGB sensor (Layer 1) — the second observation modality.

Proves the architecture's sensor-swap invariant (doc §49, §248; ARCHITECTURE
"swap the sensor, touch neither core nor the other sensor"): the SAME Layer-0
scene — the centerline frame, the shared R(s,θ) wall field, and the device
polyline — is observed through a forward-looking endoscopic camera instead of a
projective X-ray. This is the bronchoscopy / GI-endoscopy / ureteroscopy sensor:
an RGB pinhole at the instrument tip looking down the lumen.

Render model (lazy but honest): a pinhole at the tip, looking along the tip
tangent, casts one ray per pixel and marches forward until it crosses the wall
r = R(s,θ). The hit is shaded by distance (the classic endoscope head-lamp
falloff — near wall bright, far tunnel dark) times a Lambertian term, with optional
deterministic mucosal texture/fold modulation. No global illumination — enough to
stand up a useful second perception modality and demonstrate the swap.

ponytail: per-pixel python ray-march (O(nu·nv·steps), one frame.project per step)
— fine at reference resolution; vectorise the march if it ever gates a training
loop (same note as the fluoro per-env render).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lumen.core.frame import CenterlineFrame


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


@dataclass
class LuminalCamera:
    """Forward-looking pinhole at the device tip. Geometry + a minimal shading model."""
    fov_deg: float = 90.0
    nu: int = 64
    nv: int = 64
    n_steps: int = 128                        # ray-march samples
    max_dist: float | None = None             # march cap; default = centerline length
    falloff: float | None = None              # head-lamp e-folding distance; default = max_dist/3
    ambient: float = 0.15                      # floor so grazing walls aren't pure black
    tissue_color: tuple = field(default_factory=lambda: (0.80, 0.30, 0.25))  # reddish
    texture_strength: float = 0.0              # deterministic mucosal mottling, off by default
    fold_strength: float = 0.0                 # weak ring/fold shading along s, off by default

    def __post_init__(self):
        if self.n_steps <= 0:                  # else dtau = max_dist / n_steps blows up
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")

    def render(self, frame: CenterlineFrame, lumen, device_nodes,
               tip_pos=None, tip_dir=None, up=None):
        """Render the lumen interior to an (nv, nu, 3) RGB image in [0, 1].

        Camera defaults to the device tip (last node), looking along the last device
        segment; override with tip_pos / tip_dir / up. `lumen` is a LumenField (or any
        object with .gap(s, theta, r) and .eval(s, theta))."""
        nodes = np.asarray(device_nodes, float)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 2:
            raise ValueError("device_nodes must be an (N>=2, 3) polyline")
        tip = np.asarray(tip_pos, float) if tip_pos is not None else nodes[-1]
        fwd = _unit(tip_dir) if tip_dir is not None else _unit(nodes[-1] - nodes[-2])

        up_hint = np.asarray(up, float) if up is not None else frame.project(tip).e_r
        right = np.cross(fwd, up_hint)
        if np.linalg.norm(right) < 1e-6:          # up ∥ fwd: pick any perpendicular
            right = np.cross(fwd, [1.0, 0.0, 0.0])
            if np.linalg.norm(right) < 1e-6:
                right = np.cross(fwd, [0.0, 1.0, 0.0])
        right = _unit(right)
        cam_up = _unit(np.cross(right, fwd))

        max_dist = float(self.max_dist) if self.max_dist is not None else frame.length
        falloff = float(self.falloff) if self.falloff is not None else max_dist / 3.0
        half = np.tan(np.radians(self.fov_deg) / 2.0)
        us = (np.arange(self.nu) + 0.5) / self.nu * 2 - 1          # [-1, 1)
        vs = (np.arange(self.nv) + 0.5) / self.nv * 2 - 1
        dtau = max_dist / self.n_steps
        tissue = np.asarray(self.tissue_color, float)

        img = np.zeros((self.nv, self.nu, 3), float)
        for iv in range(self.nv):
            for iu in range(self.nu):
                d = _unit(fwd + (us[iu] * half) * right + (vs[iv] * half) * cam_up)
                hit, e_r, s_hit, th_hit = self._march(frame, lumen, tip, d, dtau, max_dist)
                if hit is None:                                    # exits the lumen end -> deep dark
                    continue
                shade = np.exp(-hit / falloff)                     # head-lamp falloff
                lamb = max(0.0, float(np.dot(d, e_r)))             # wall faces the (co-located) lamp
                detail = self._surface_detail(s_hit, th_hit)
                img[iv, iu] = shade * (self.ambient + (1 - self.ambient) * lamb) * detail * tissue
        return np.clip(img, 0.0, 1.0)

    def _surface_detail(self, s, theta):
        """Deterministic wall detail. Defaults to 1.0 for the original smooth tunnel."""
        detail = 1.0
        if self.texture_strength:
            tex = (0.55 * np.sin(0.73 * s + 5.1 * theta)
                   + 0.35 * np.sin(1.91 * s - 2.7 * theta)
                   + 0.10 * np.sin(3.7 * theta))
            detail += self.texture_strength * tex
        if self.fold_strength:
            fold = 0.5 + 0.5 * np.sin(0.42 * s + 0.8 * np.sin(theta))
            detail *= 1.0 + self.fold_strength * fold
        return max(0.0, float(detail))

    @staticmethod
    def _march(frame, lumen, origin, d, dtau, max_dist):
        """March origin+τd outward; return (distance, e_r) at the first wall crossing
        (gap = R - r <= 0), or (None, None, None, None) if it leaves the lumen end first."""
        tau = dtau                                                 # start just ahead of the tip
        while tau <= max_dist:
            pr = frame.project(origin + tau * d)
            if lumen.gap(pr.s, pr.theta, pr.r) <= 0.0:            # crossed the wall
                return tau, pr.e_r, pr.s, pr.theta
            tau += dtau
        return None, None, None, None


if __name__ == "__main__":  # self-check: tunnel shape + stenosis brightens the view
    from lumen.assets import procedural

    def _tip_setup(asset):
        pts, lumen = asset.edge_arrays(asset.edges[0])
        pts = np.asarray(pts)
        frame = CenterlineFrame(pts)
        dev = np.stack([pts[1], pts[3]])                          # two nodes near the inlet -> forward dir
        return frame, lumen, dev

    asset = procedural.straight_tube(80.0, 4.0)
    frame, lumen, dev = _tip_setup(asset)
    cam = LuminalCamera(nu=24, nv=24, n_steps=64)
    img = cam.render(frame, lumen, dev)
    assert img.shape == (24, 24, 3) and img.min() >= 0 and img.max() <= 1

    # straight tube down the axis: off-axis rays hit the side wall NEARER than the
    # far end, so the edges are brighter than the dead-ahead centre (the tunnel look).
    gray = img.mean(axis=2)
    c = gray[10:14, 10:14].mean()
    edge = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]]).mean()
    assert edge > c, f"tunnel edge ({edge:.3f}) should be brighter than centre ({c:.3f})"

    # a stenosis ahead pulls the wall closer -> brighter mean than a wide tube.
    sten = procedural.stenotic_tube(80.0, 4.0, severity=0.7)
    fs, ls, ds = _tip_setup(sten)
    img_s = LuminalCamera(nu=24, nv=24, n_steps=64).render(fs, ls, ds)
    assert img_s.mean() > img.mean(), "a narrowing ahead should brighten the view"
    print("luminal self-check ok")
