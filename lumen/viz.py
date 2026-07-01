"""Schematic 2-D viewer for lumen navigation scenes.

A dependency-light, headless renderer: it rasterizes the vessel lumen, the target
band, and the guidewire onto an RGB canvas each step, so a rollout becomes an
inspectable animation without a GPU, a display, or a 3-D engine. Pure NumPy; frames
are written by the stdlib AVI/PNG writers in ``lumen.sensors.preview``.

This is the headless equivalent of an interactive scene viewer: `lumen play` drives
a scene with a chosen policy and writes `<out>.avi` + `<out>.png`, reporting the same
tip-reach / wall-safety numbers the benchmark scores.
"""

from __future__ import annotations

import numpy as np

# colors (RGB uint8)
_BG = (10, 12, 18)
_WALL = (196, 72, 66)          # vessel wall
_LUMEN = (26, 30, 44)          # inside-the-lumen fill
_DEVICE = (86, 230, 236)       # guidewire (cyan)
_TIP_SAFE = (120, 240, 140)    # tip, not touching wall
_TIP_HIT = (255, 90, 90)       # tip at/over the wall-safety threshold
_TARGET = (245, 210, 90)       # target band


def _principal_axes(pts: np.ndarray) -> tuple[int, int]:
    """The two world axes with the largest extent — the plane that best shows the scene."""
    ext = pts.max(0) - pts.min(0)
    order = np.argsort(ext)[::-1]
    return int(order[0]), int(order[1])


def _arclength(pts: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


class _Canvas:
    def __init__(self, h: int, w: int, bg=_BG):
        self.h, self.w = h, w
        self.img = np.empty((h, w, 3), np.uint8)
        self.img[:] = np.array(bg, np.uint8)

    def disk(self, y: float, x: float, rad: float, color) -> None:
        r = int(np.ceil(rad))
        yi, xi = int(round(y)), int(round(x))
        y0, y1 = max(0, yi - r), min(self.h, yi + r + 1)
        x0, x1 = max(0, xi - r), min(self.w, xi + r + 1)
        if y0 >= y1 or x0 >= x1:
            return
        ys, xs = np.ogrid[y0:y1, x0:x1]
        mask = (ys - y) ** 2 + (xs - x) ** 2 <= rad ** 2
        self.img[y0:y1, x0:x1][mask] = np.array(color, np.uint8)

    def polyline(self, ys, xs, color, width: float = 1.5) -> None:
        ys, xs = np.asarray(ys, float), np.asarray(xs, float)
        for i in range(len(ys) - 1):
            n = int(max(2, np.hypot(ys[i + 1] - ys[i], xs[i + 1] - xs[i])))
            for t in np.linspace(0, 1, n):
                self.disk(ys[i] + t * (ys[i + 1] - ys[i]),
                          xs[i] + t * (xs[i + 1] - xs[i]), width, color)


def _projector(all_pts2d: np.ndarray, size: int, pad: float):
    """Return a function world-2D -> pixel (row, col), equal aspect, `pad` fractional margin."""
    lo, hi = all_pts2d.min(0), all_pts2d.max(0)
    span = np.maximum(hi - lo, 1e-6)
    scale = (1 - 2 * pad) * size / span.max()
    off = 0.5 * (size - scale * span)

    def to_px(p2d: np.ndarray) -> np.ndarray:
        q = (p2d - lo) * scale + off
        col = q[..., 0]
        row = size - q[..., 1]              # flip y so +axis points up
        return np.stack([row, col], axis=-1)

    return to_px


def _centerline_and_R(env):
    """Centerline points (n,3) and per-station wall radius R(s) (n,) for the active scene."""
    frame = getattr(env, "frame", None) or getattr(env, "route_frame", None)
    pts = np.asarray(frame.points)
    lumen = getattr(env, "lumen", None)
    if lumen is not None:
        R = np.asarray(lumen.R, float).reshape(len(np.asarray(lumen.R)), -1).mean(1)
        if len(R) != len(pts):                 # resample to centerline stations
            R = np.interp(np.linspace(0, 1, len(pts)), np.linspace(0, 1, len(R)), R)
    else:
        R = np.full(len(pts), float(env.R))
    return pts, R, frame


def render_frame(env, size: int = 480, pad: float = 0.12) -> np.ndarray:
    """Rasterize the current scene + guidewire state to an HxWx3 uint8 frame."""
    pts, R, frame = _centerline_and_R(env)
    a0, a1 = _principal_axes(pts)
    c2d = pts[:, [a0, a1]]

    # in-plane normals for the wall offset
    tang = np.gradient(c2d, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    wall_p = c2d + R[:, None] * normal
    wall_m = c2d - R[:, None] * normal

    dev = np.asarray(env.sim.body_positions())[:, [a0, a1]]

    # target point at arc-length target_s
    s = _arclength(pts)
    ts = float(getattr(env, "target_s", s[-1]))
    tgt2d = np.array([np.interp(ts, s, c2d[:, 0]), np.interp(ts, s, c2d[:, 1])])

    to_px = _projector(np.concatenate([wall_p, wall_m, dev, tgt2d[None]]), size, pad)
    cv, cvR = _Canvas(size, size), float(env.R)
    wp, wm, cc = to_px(wall_p), to_px(wall_m), to_px(c2d)

    # lumen fill (between the two walls, faint) then wall lines
    for i in range(len(cc) - 1):
        for a, b in ((wp, wm), (wm, wp)):
            cv.polyline([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], _LUMEN, 1.0)
    cv.polyline(wp[:, 0], wp[:, 1], _WALL, 2.0)
    cv.polyline(wm[:, 0], wm[:, 1], _WALL, 2.0)

    # target band
    tp = to_px(tgt2d)
    cv.disk(tp[0], tp[1], 8, _TARGET)
    cv.disk(tp[0], tp[1], 4, _BG)

    # guidewire + tip (red if the tip radius has reached the wall-safety threshold)
    dp = to_px(dev)
    cv.polyline(dp[:, 0], dp[:, 1], _DEVICE, 2.5)
    rmax = float(env.sim.node_radii().max())
    tip_color = _TIP_HIT if rmax >= cvR else _TIP_SAFE
    cv.disk(dp[-1, 0], dp[-1, 1], 5, tip_color)
    return cv.img


def _policy(name):
    name = name or "forward"
    # a trained policy saved by `lumen train` (theta under key 'theta', or the first array)
    if isinstance(name, str) and name.endswith(".npz"):
        from lumen.rl.cem import make_policy
        data = np.load(name)
        theta = data["theta"] if "theta" in data else data[data.files[0]]
        return make_policy(theta)
    name = name.lower()
    if name in ("forward", "advance"):
        return lambda obs: np.array([1.0], np.float32)
    if name == "zero":
        return lambda obs: np.array([0.0], np.float32)
    if name == "random":
        rng = np.random.default_rng(0)
        return lambda obs: rng.uniform(-1, 1, size=1).astype(np.float32)
    raise ValueError(f"unknown policy {name!r} (forward|zero|random|*.npz)")


def play(scene: str = "tube", policy="forward", steps: int = 60, seed: int = 0,
         out: str | None = None, size: int = 480, perforation: float = 0.3,
         env=None) -> dict:
    """Roll out a scene under a policy, render each step, and (if `out`) write an
    animation. Returns a summary with the same tip-reach / wall-safety fields the
    benchmark reports.

    scene: 'tube' | 'stenotic' | 'tree'  (ignored if `env` is passed).
    policy: name ('forward'|'zero'|'random') or a callable obs->action.
    """
    if env is None:
        from lumen.envs import registration as reg
        factory = {"tube": reg.make_nav_tube, "stenotic": reg.make_nav_stenotic,
                   "tree": reg.make_tree_nav}.get(scene)
        if factory is None:
            raise ValueError(f"unknown scene {scene!r} (tube|stenotic|tree)")
        env = factory()
    pol = policy if callable(policy) else _policy(policy)

    obs, _ = env.reset(seed=seed)
    frames = [render_frame(env, size=size)]
    max_pen, success = 0.0, False
    used = 0
    for used in range(1, steps + 1):
        obs, _, terminated, truncated, info = env.step(pol(obs))
        max_pen = max(max_pen, max(0.0, float(env.sim.node_radii().max()) - float(env.R)))
        frames.append(render_frame(env, size=size))
        if info.get("success"):
            success = True
        if terminated or truncated:
            break

    summary = {"scene": scene, "steps": used, "frames": len(frames),
               "success": bool(success), "max_pen": round(max_pen, 4),
               "safe": bool(max_pen <= perforation),
               "safe_success": bool(success and max_pen <= perforation)}
    if out is not None:
        from pathlib import Path

        from lumen.sensors.preview import write_avi, write_png
        out = Path(out)
        stem = out.with_suffix("") if out.suffix else out
        write_avi(stem.with_suffix(".avi"), frames)
        write_png(stem.with_suffix(".png"), frames[-1])
        summary["avi"] = str(stem.with_suffix(".avi"))
        summary["png"] = str(stem.with_suffix(".png"))
    return summary


if __name__ == "__main__":   # tiny self-check: renders and reports without writing files
    for sc in ("tube", "stenotic"):
        s = play(sc, steps=8, size=160)
        assert s["frames"] == s["steps"] + 1, s
        assert 0 <= s["max_pen"] < 100, s
        print(sc, s)
    print("viz self-check ok")
