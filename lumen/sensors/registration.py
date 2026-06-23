"""L1.1 — couple Layer 0 → renderer and invert it: 2-D/3-D registration (doc §3.6).

The forward map device-pose → node polyline → μ volume → synthetic fluoro is one
pipeline; registration runs it in inverse, recovering the rigid pose that best
reproduces a target image (image-space loss). This is the first of the doc's three
imaging-loop capabilities and the scaffold the device-as-sensor loop (L1.2) reuses.

Identifiability note (the doc's standing caveat, §3.6): a single projection has a
depth/out-of-plane ambiguity — in-plane pose is recovered well, out-of-plane is
under-determined. Biplanar (two C-arms) resolves it; pass two views to `register`.
"""

from __future__ import annotations

import numpy as np

from lumen.sensors._optim import fd_minimize


def _rodrigues(rvec):
    ang = float(np.linalg.norm(rvec))
    if ang < 1e-12:
        return np.eye(3)
    k = rvec / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def apply_se3(nodes, pose):
    """Rigid pose on a node polyline: rotate (axis-angle pose[3:6]) about the centroid,
    then translate (pose[0:3]). Centroid-relative so rotation doesn't fling the device."""
    if len(pose) != 6:                               # L3: [tx,ty,tz, rx,ry,rz]
        raise ValueError(f"pose must have 6 elements, got {len(pose)}")
    nodes = np.asarray(nodes, float)
    R = _rodrigues(np.asarray(pose[3:6], float))
    c = nodes.mean(0)
    return (nodes - c) @ R.T + c + np.asarray(pose[0:3], float)


def _render_views(nodes, sensor, carms):
    return [sensor.render(nodes, carm=c)[0] for c in carms]


def image_loss(nodes, targets, sensor, carms):
    """Mean-squared image error summed over views."""
    return float(sum(np.mean((a - t) ** 2) for a, t in zip(_render_views(nodes, sensor, carms), targets)))


def register(targets, device_nodes, sensor, carms, init_pose=None, iters=40, lr=0.4):
    """Recover the device rigid pose from target fluoro image(s).

    `targets`/`carms` are lists (1 = mono, 2 = biplanar). Returns (pose, history).
    `pose` is [tx,ty,tz, rx,ry,rz]; apply with apply_se3."""
    carms = [carms] if hasattr(carms, "rays") else list(carms)       # bare CArm -> [CArm]
    targets = [targets] if np.ndim(targets) == 2 else list(targets)  # bare image -> [image]
    if len(carms) != len(targets):                                   # H1: no silent zip-truncation
        raise ValueError(f"{len(carms)} carms vs {len(targets)} targets")
    device_nodes = np.asarray(device_nodes, float)
    span = float(np.ptp(device_nodes, axis=0).max()) + 1e-9
    x0 = np.zeros(6) if init_pose is None else np.asarray(init_pose, float)
    scale = np.array([0.15 * span] * 3 + [0.25, 0.25, 0.25])      # mm vs rad conditioning

    def loss(p):
        return image_loss(apply_se3(device_nodes, p), targets, sensor, carms)

    return fd_minimize(loss, x0, scale, iters=iters, lr=lr)
