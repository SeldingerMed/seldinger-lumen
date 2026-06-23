"""Device-state perception from fluoroscopy (Layer 1, doc §4.2).

The image-based control loop needs the device's state read FROM the image, not from
privileged simulator state. This is the minimal front-end: detect the radio-opaque
device's leading-tip pixel in a DRR/fluoro frame (the device is the high-attenuation
region). Learned shape-recovery nets (DeepWire/SplineFormer, §4.2) are the upgrade
behind this same interface; this thresholding tip-finder is the dependency-light stand-in.
"""

from __future__ import annotations

import numpy as np


def detect_device_tip(image, thresh_frac=0.5, leading="max_v"):
    """Leading device-tip pixel (u, v) in a DRR line-integral / attenuation image.

    The device is the bright (high-A) region; the inserted tip is its extreme along the
    projected vessel direction — by default the largest detector-v. Returns
    (u, v, present): (u,v) pixel floats and a presence flag (0 if no device detected)."""
    A = np.asarray(image, float)
    if A.max() <= 0:
        return 0.0, 0.0, 0.0
    ys, xs = np.where(A > thresh_frac * A.max())
    if len(ys) == 0:
        return 0.0, 0.0, 0.0
    i = np.argmax(ys) if leading == "max_v" else np.argmin(ys)
    return float(xs[i]), float(ys[i]), 1.0


def device_centroid(image, thresh_frac=0.5):
    """Attenuation-weighted centroid (u, v) of the device (a robustness feature)."""
    A = np.asarray(image, float)
    m = A > thresh_frac * A.max() if A.max() > 0 else np.zeros_like(A, bool)
    if not m.any():
        return 0.0, 0.0
    ys, xs = np.where(m)
    wts = A[ys, xs]
    return float((xs * wts).sum() / wts.sum()), float((ys * wts).sum() / wts.sum())
