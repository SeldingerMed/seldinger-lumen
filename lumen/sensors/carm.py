"""C-arm fluoroscopy geometry (Layer 1, doc §3.6, §4.1).

A divergent-beam X-ray camera: a point source S and a flat detector. Unlike an RGB
pinhole, the X-ray "camera" is a *source* that casts diverging rays through the scene
onto the detector — the DRR (digitally reconstructed radiograph) integrates
attenuation along those rays. This module just holds the geometry and emits one ray
(origin = source, direction → each detector pixel centre) per pixel.

Mono view for now; biplanar (two C-arms) is a later milestone where the device-as-
sensor inverse problem needs the extra view for identifiability (doc §3.6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


@dataclass
class CArm:
    """Source–detector geometry. The detector is centred at ``detector_center`` with
    in-plane axes (u right, v up) inferred from ``view_dir`` (source→detector) and
    ``up``; ``width``/``height`` are physical detector extents, ``nu``/``nv`` pixels."""
    source: np.ndarray                       # X-ray source point (3,)
    detector_center: np.ndarray              # detector centre (3,)
    up: np.ndarray = None                    # world up hint for the detector v-axis
    width: float = 60.0
    height: float = 60.0
    nu: int = 128
    nv: int = 128

    @classmethod
    def looking_at(cls, target, distance=120.0, axis=(1.0, 0.0, 0.0), sdd=240.0,
                   up=(0.0, 0.0, 1.0), **kw):
        """Place a C-arm viewing `target` along `axis`: source at -axis·distance, the
        detector on the far side so the beam passes through the scene. `sdd` is the
        source-to-detector distance."""
        target = np.asarray(target, float)
        a = _unit(axis)
        source = target - a * distance
        detector_center = source + a * sdd
        return cls(source=source, detector_center=detector_center, up=np.asarray(up, float), **kw)

    def axes(self):
        """Detector orthonormal in-plane axes (u, v) and inward normal n (source→det)."""
        n = _unit(np.asarray(self.detector_center) - np.asarray(self.source))
        up = self.up if self.up is not None else np.array([0.0, 0.0, 1.0])
        u = _unit(np.cross(n, up))
        if np.linalg.norm(np.cross(n, up)) < 1e-6:        # degenerate: up ∥ n
            u = _unit(np.cross(n, np.array([1.0, 0.0, 0.0])))
        v = _unit(np.cross(u, n))
        return u, v, n

    def pixel_points(self):
        """World-space centres of every detector pixel, shape (nv, nu, 3)."""
        u, v, _ = self.axes()
        us = (np.arange(self.nu) + 0.5) / self.nu - 0.5    # in [-0.5, 0.5)
        vs = (np.arange(self.nv) + 0.5) / self.nv - 0.5
        gu, gv = np.meshgrid(us * self.width, vs * self.height)   # (nv, nu)
        return (np.asarray(self.detector_center)[None, None, :]
                + gu[..., None] * u[None, None, :]
                + gv[..., None] * v[None, None, :])

    def rays(self):
        """(origin (3,), directions (nv,nu,3) unit) from the source to each pixel."""
        src = np.asarray(self.source, float)
        pts = self.pixel_points()
        d = pts - src[None, None, :]
        d /= (np.linalg.norm(d, axis=2, keepdims=True) + 1e-12)
        return src, d
