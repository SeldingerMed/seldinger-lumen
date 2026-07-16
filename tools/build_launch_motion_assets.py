"""Build motion assets for the Lumen launch video.

The clips are generated from Lumen procedural geometry and reduced-order state.
They deliberately avoid the old 2-D rollout artifact where the guidewire appeared
to spin around the vessel. The navigation clip advances monotonically along a
centerline route into a branch while the camera orbits the scene.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np


BG = np.array([7, 10, 14], dtype=np.uint8)
GRID = np.array([18, 29, 38], dtype=np.uint8)
LINE = np.array([68, 84, 104], dtype=np.uint8)
VESSEL = np.array([82, 25, 29], dtype=np.uint8)
VESSEL_EDGE = np.array([225, 92, 88], dtype=np.uint8)
WIRE = np.array([92, 232, 238], dtype=np.uint8)
WIRE_CORE = np.array([230, 255, 255], dtype=np.uint8)
TARGET = np.array([244, 205, 84], dtype=np.uint8)
GREEN = np.array([112, 211, 141], dtype=np.uint8)
AMBER = np.array([236, 185, 88], dtype=np.uint8)
VIOLET = np.array([156, 132, 232], dtype=np.uint8)


def _canvas(h: int, w: int) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = BG
    for x in range(0, w, 80):
        img[:, x:x + 1] = GRID
    for y in range(0, h, 80):
        img[y:y + 1, :] = GRID
    return img


def _blend_disk(img: np.ndarray, y: float, x: float, r: float, color: np.ndarray, alpha: float = 1.0) -> None:
    h, w = img.shape[:2]
    rr = max(1, int(math.ceil(r)))
    yi, xi = int(round(y)), int(round(x))
    y0, y1 = max(0, yi - rr), min(h, yi + rr + 1)
    x0, x1 = max(0, xi - rr), min(w, xi + rr + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (yy - y) ** 2 + (xx - x) ** 2
    mask = d2 <= r * r
    if not np.any(mask):
        return
    patch = img[y0:y1, x0:x1]
    if alpha >= 0.999:
        patch[mask] = color
    else:
        patch[mask] = (patch[mask].astype(float) * (1.0 - alpha) + color.astype(float) * alpha).astype(np.uint8)


def _polyline(img: np.ndarray, pts: np.ndarray, color: np.ndarray, width: float, alpha: float = 1.0) -> None:
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(2, int(np.linalg.norm(b - a) * 1.7))
        for t in np.linspace(0.0, 1.0, n):
            p = a + t * (b - a)
            _blend_disk(img, p[1], p[0], width, color, alpha)


def _resample_path(points: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    q = np.linspace(0.0, s[-1], n)
    out = np.column_stack([np.interp(q, s, pts[:, ax]) for ax in range(pts.shape[1])])
    return out


def _rotation(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=float)
    return rz @ rx @ ry


def _project(points: np.ndarray, rot: np.ndarray, w: int, h: int, zoom: float = 6.4) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(points, dtype=float)
    q = p @ rot.T
    z = q[:, 2]
    # Mild perspective; enough depth to feel 3-D without distorting the anatomy.
    f = zoom * min(w, h)
    denom = np.maximum(420.0 + z, 80.0)
    x = w * 0.5 + f * q[:, 0] / denom
    y = h * 0.54 - f * q[:, 1] / denom
    return np.column_stack([x, y]), z


def _read_tree_geometry():
    from lumen.assets import procedural

    asset = procedural.tortuous_tree(radius=4.0)
    tubes: list[tuple[np.ndarray, np.ndarray]] = []
    edge_points: dict[str, np.ndarray] = {}
    for edge in asset.edges:
        pts, lf = asset.edge_arrays(edge)
        r = np.asarray(lf.R, dtype=float).reshape(len(pts), -1).mean(axis=1)
        sampled = _resample_path(np.asarray(pts, dtype=float), 120)
        edge_points[edge.id] = sampled
        tubes.append((sampled, np.interp(np.linspace(0, 1, 120), np.linspace(0, 1, len(r)), r)))
    # Route: inlet trunk -> apex -> stenotic target branch. Build it by named,
    # connected edges so the visible wire never jumps backward across anatomy.
    route_ids = ("trunk_in", "trunk_mid", "right_stenotic")
    route = np.concatenate(
        [edge_points[route_ids[0]]] + [edge_points[edge_id][1:] for edge_id in route_ids[1:]],
        axis=0,
    )
    route = _resample_path(route, 220)
    all_pts = np.concatenate([p for p, _ in tubes] + [route], axis=0)
    center = all_pts.mean(axis=0)
    tubes = [(p - center, r) for p, r in tubes]
    route = route - center
    return tubes, route


def _draw_vessel(img: np.ndarray, tubes: list[tuple[np.ndarray, np.ndarray]], rot: np.ndarray) -> None:
    h, w = img.shape[:2]
    samples = []
    for pts, radii in tubes:
        pix, z = _project(pts, rot, w, h)
        dz = z - z.min()
        shade = 0.72 + 0.28 * dz / max(float(dz.max()), 1e-6)
        for i in range(len(pix)):
            samples.append((z[i], pix[i], radii[i], shade[i]))
    for _, p, r, shade in sorted(samples, key=lambda x: x[0]):
        rad = max(5.0, float(r) * 5.6)
        fill = np.clip(VESSEL.astype(float) * shade, 0, 255).astype(np.uint8)
        _blend_disk(img, p[1], p[0], rad, fill, 0.78)
    for pts, _ in tubes:
        pix, z = _project(pts, rot, w, h)
        depth = (0.82 + 0.18 * (z - z.min()) / max(float(np.ptp(z)), 1e-6))[:, None]
        order_color = np.clip(
            VESSEL_EDGE.astype(float)[None, :] * depth,
            0,
            255,
        ).astype(np.uint8)
        # Draw as short segments so the color can carry depth.
        for i in range(len(pix) - 1):
            _polyline(img, pix[i:i + 2], order_color[i], 1.7, 0.88)


def _draw_wire(img: np.ndarray, route: np.ndarray, rot: np.ndarray, progress: float) -> tuple[float, float]:
    h, w = img.shape[:2]
    n = max(8, int(progress * (len(route) - 1)))
    visible = route[:n]
    pix, z = _project(visible, rot, w, h)
    _polyline(img, pix, np.array([4, 28, 38], np.uint8), 8.0, 0.82)
    _polyline(img, pix, WIRE, 3.6, 1.0)
    _polyline(img, pix, WIRE_CORE, 1.2, 1.0)
    tip = pix[-1]
    _blend_disk(img, tip[1], tip[0], 8.0, GREEN, 1.0)
    _blend_disk(img, tip[1], tip[0], 3.5, WIRE_CORE, 1.0)
    return float(tip[0]), float(tip[1])


def _draw_target(img: np.ndarray, route: np.ndarray, rot: np.ndarray) -> None:
    h, w = img.shape[:2]
    pix, _ = _project(route[-1:], rot, w, h)
    x, y = pix[0]
    _blend_disk(img, y, x, 16, TARGET, 1.0)
    _blend_disk(img, y, x, 9, BG, 1.0)
    _blend_disk(img, y, x, 4, TARGET, 1.0)


def _overlay_ui(img: np.ndarray, frame: int, total: int, tip_xy: tuple[float, float], label: str) -> None:
    h, w = img.shape[:2]
    x0, y0 = 52, 52
    img[y0:y0 + 2, x0:w - x0] = LINE
    img[h - y0:h - y0 + 2, x0:w - x0] = LINE
    # Progress bar and small tracking reticle.
    p = frame / max(total - 1, 1)
    img[h - 104:h - 98, x0:x0 + int((w - 2 * x0) * p)] = CYAN = np.array([101, 215, 225], dtype=np.uint8)
    tx, ty = tip_xy
    _polyline(img, np.array([[tx - 24, ty], [tx - 9, ty], [tx + 9, ty], [tx + 24, ty]]), CYAN, 1.2, 0.85)
    _polyline(img, np.array([[tx, ty - 24], [tx, ty - 9], [tx, ty + 9], [tx, ty + 24]]), CYAN, 1.2, 0.85)
    # Text-free telemetry blocks: avoids needing font deps, still reads as motion graphics.
    for i, color in enumerate((GREEN, TARGET, WIRE)):
        xx = w - 340 + i * 94
        img[82:132, xx:xx + 70] = (img[82:132, xx:xx + 70].astype(float) * 0.55).astype(np.uint8)
        img[86:92, xx + 6:xx + 64] = color
        img[104:110, xx + 6:xx + 34 + int(24 * math.sin(frame * 0.08 + i) ** 2)] = color
    if label:
        # Label represented as a cyan tag line to keep generated frames dependency-light.
        img[150:154, 62:62 + min(620, 18 * len(label))] = np.array([101, 215, 225], dtype=np.uint8)


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    if not frames:
        raise ValueError("no frames")
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def build_navigation_3d(out: Path, *, seconds: float = 5.0, fps: int = 30, size=(720, 1280)) -> None:
    h, w = size
    tubes, route = _read_tree_geometry()
    total = int(seconds * fps)
    frames = []
    for i in range(total):
        t = i / max(total - 1, 1)
        # Smooth advance, no overshoot, no wrap.
        progress = 0.08 + 0.90 * (3 * t * t - 2 * t * t * t)
        yaw = math.radians(-18 + 30 * t)
        pitch = math.radians(16 + 4 * math.sin(2 * math.pi * t))
        rot = _rotation(yaw, pitch, math.radians(-3))
        img = _canvas(h, w)
        _draw_vessel(img, tubes, rot)
        _draw_target(img, route, rot)
        tip = _draw_wire(img, route, rot, progress)
        _overlay_ui(img, i, total, tip, "safe branch route")
        frames.append(img)
    _write_mp4(out / "real-tree-navigation-3d.mp4", frames, fps=fps)


def build_physics_3d(out: Path, *, seconds: float = 4.4, fps: int = 30, size=(720, 1280)) -> None:
    h, w = size
    tubes, route = _read_tree_geometry()
    route = route[40:150]
    total = int(seconds * fps)
    frames = []
    for i in range(total):
        t = i / max(total - 1, 1)
        img = _canvas(h, w)
        rot = _rotation(math.radians(26 + 64 * t), math.radians(22), math.radians(4))
        small_tubes = [(route, np.full(len(route), 4.6))]
        _draw_vessel(img, small_tubes, rot)
        pix, _ = _project(route, rot, w, h)
        # Animated flow pulses inside the vessel.
        for k in range(9):
            u = (t * 1.25 + k / 9.0) % 1.0
            idx = min(len(pix) - 1, max(0, int(u * (len(pix) - 1))))
            _blend_disk(img, pix[idx, 1], pix[idx, 0], 10 + 5 * math.sin(u * math.pi), AMBER, 0.45)
        # Stentriever/flow diverter scaffold: periodic rings around the route.
        for idx in range(16, len(pix) - 16, 12):
            col = GREEN if idx % 24 else VIOLET
            _blend_disk(img, pix[idx, 1], pix[idx, 0], 8.5, col, 0.95)
            _blend_disk(img, pix[idx, 1], pix[idx, 0], 4.0, BG, 1.0)
        _polyline(img, pix, WIRE, 1.5, 0.65)
        _overlay_ui(img, i, total, (float(pix[-1, 0]), float(pix[-1, 1])), "flow clot device")
        frames.append(img)
    _write_mp4(out / "real-physics-3d.mp4", frames, fps=fps)


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    build_navigation_3d(out)
    build_physics_3d(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="launch/video/lumen-launch-video/assets")
    args = parser.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()
