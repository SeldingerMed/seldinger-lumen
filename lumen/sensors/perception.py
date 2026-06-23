"""Device-state perception from fluoroscopy (Layer 1, doc §4.2).

The image-based control loop needs the device's state read FROM the image, not from
privileged simulator state. This is the minimal front-end: detect the radio-opaque
device's leading-tip pixel in a DRR/fluoro frame (the device is the high-attenuation
region). This thresholding tip-finder is the dependency-light stand-in; learned
shape-recovery nets (DeepWire/SplineFormer, §4.2) are the upgrade — they return a full
spline, so a richer `detect_device(image) -> spline` will sit behind a *compatible*
(not identical) interface, of which the tip is one readout.
"""

from __future__ import annotations

import numpy as np


def detect_device_tip(image, thresh_frac=0.5, leading="max_v"):
    """Leading device-tip pixel (u, v) in a DRR line-integral / attenuation image.

    The device is the bright (high-A) region; the inserted tip is its extreme along the
    projected vessel direction — `leading` selects which detector-v extreme (the caller
    picks it from the vessel's projected direction). Returns (u, v, present): (u,v)
    pixel floats and a presence flag (0 if no device detected)."""
    if leading not in ("max_v", "min_v"):                # L4: catch typos, don't silently pick min
        raise ValueError(f"leading must be 'max_v' or 'min_v', got {leading!r}")
    A = np.nan_to_num(np.asarray(image, float), nan=0.0, posinf=0.0, neginf=0.0)  # M2
    if A.max() <= 0:
        return 0.0, 0.0, 0.0
    ys, xs = np.where(A > thresh_frac * A.max())
    if len(ys) == 0:
        return 0.0, 0.0, 0.0
    i = np.argmax(ys) if leading == "max_v" else np.argmin(ys)
    return float(xs[i]), float(ys[i]), 1.0


def device_centroid(image, thresh_frac=0.5):
    """Attenuation-weighted centroid (u, v) of the device (a robustness feature)."""
    A = np.nan_to_num(np.asarray(image, float), nan=0.0, posinf=0.0, neginf=0.0)  # M2
    m = A > thresh_frac * A.max() if A.max() > 0 else np.zeros_like(A, bool)
    if not m.any():
        return 0.0, 0.0
    ys, xs = np.where(m)
    wts = A[ys, xs]
    return float((xs * wts).sum() / wts.sum()), float((ys * wts).sum() / wts.sum())
