"""Build launch visuals from real Lumen simulator/render outputs.

The output is intentionally capture-first: every clinical-looking panel starts
from Lumen objects, environments, sensors, or solver-side reduced models. Text
and marketing composition belong in the page/video layer, not burned over the
simulation frames.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from lumen.sensors.preview import write_avi, write_png


BG = np.array([8, 10, 14], dtype=np.uint8)
PANEL = np.array([15, 19, 25], dtype=np.uint8)
LINE = np.array([80, 92, 108], dtype=np.uint8)
CYAN = np.array([101, 215, 225], dtype=np.uint8)
GREEN = np.array([112, 211, 141], dtype=np.uint8)
AMBER = np.array([236, 185, 88], dtype=np.uint8)
RED = np.array([218, 91, 94], dtype=np.uint8)
VIOLET = np.array([156, 132, 232], dtype=np.uint8)


def _u8(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype == np.uint8:
        return arr
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo >= 0.0 and hi <= 1.0:
        out = arr * 255.0
    elif hi > lo:
        out = (arr - lo) / (hi - lo) * 255.0
    else:
        out = np.zeros_like(arr)
    return np.clip(np.nan_to_num(out), 0, 255).astype(np.uint8)


def rgb(a: np.ndarray) -> np.ndarray:
    out = _u8(a)
    if out.ndim == 2:
        out = np.repeat(out[:, :, None], 3, axis=2)
    return out


def fit(img: np.ndarray, h: int, w: int, fill=PANEL) -> np.ndarray:
    src = rgb(img)
    ih, iw = src.shape[:2]
    scale = min(h / ih, w / iw)
    nh, nw = max(1, int(round(ih * scale))), max(1, int(round(iw * scale)))
    yy = np.clip((np.arange(nh) / scale).astype(int), 0, ih - 1)
    xx = np.clip((np.arange(nw) / scale).astype(int), 0, iw - 1)
    resized = src[yy][:, xx]
    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:] = fill
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def canvas(h: int, w: int, color=BG) -> np.ndarray:
    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:] = color
    return out


def paste(dst: np.ndarray, src: np.ndarray, y: int, x: int) -> None:
    h, w = src.shape[:2]
    dst[y:y + h, x:x + w] = src


def border(img: np.ndarray, color=LINE, width: int = 2) -> np.ndarray:
    out = img.copy()
    out[:width, :, :] = color
    out[-width:, :, :] = color
    out[:, :width, :] = color
    out[:, -width:, :] = color
    return out


def polyline(img: np.ndarray, pts: np.ndarray, color, width: int = 2) -> None:
    pts = np.asarray(pts, dtype=float)
    for a, b in zip(pts[:-1], pts[1:]):
        dist = max(2, int(np.linalg.norm(b - a)))
        for t in np.linspace(0.0, 1.0, dist):
            p = a + t * (b - a)
            disk(img, p[1], p[0], width, color)


def disk(img: np.ndarray, y: float, x: float, r: float, color) -> None:
    h, w = img.shape[:2]
    rr = int(math.ceil(r))
    yi, xi = int(round(y)), int(round(x))
    y0, y1 = max(0, yi - rr), min(h, yi + rr + 1)
    x0, x1 = max(0, xi - rr), min(w, xi + rr + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (yy - y) ** 2 + (xx - x) ** 2 <= r ** 2
    img[y0:y1, x0:x1][mask] = color


def chart(series: list[tuple[np.ndarray, np.ndarray, np.ndarray]], *, h=420, w=720) -> np.ndarray:
    out = canvas(h, w, PANEL)
    margin = 44
    xvals = np.concatenate([s[0] for s in series])
    yvals = np.concatenate([s[1] for s in series])
    xmin, xmax = float(xvals.min()), float(xvals.max())
    ymin, ymax = float(yvals.min()), float(yvals.max())
    pad = max((ymax - ymin) * 0.08, 1e-6)
    ymin -= pad
    ymax += pad
    # grid
    for k in range(5):
        y = margin + k * (h - 2 * margin) / 4
        out[int(y):int(y) + 1, margin:w - margin] = np.array([38, 46, 58], np.uint8)
    out[h - margin:h - margin + 2, margin:w - margin] = LINE
    out[margin:h - margin, margin:margin + 2] = LINE

    def map_xy(x, y):
        px = margin + (x - xmin) / max(xmax - xmin, 1e-9) * (w - 2 * margin)
        py = h - margin - (y - ymin) / max(ymax - ymin, 1e-9) * (h - 2 * margin)
        return np.stack([px, py], axis=1)

    for x, y, color in series:
        pts = map_xy(x, y)
        polyline(out, pts, color, 3)
        for px, py in pts[:: max(1, len(pts) // 12)]:
            disk(out, py, px, 3.5, color)
    return border(out)


def montage(images: list[np.ndarray], rows: int, cols: int, cell_h: int, cell_w: int,
            gap: int = 18) -> np.ndarray:
    out = canvas(rows * cell_h + (rows + 1) * gap, cols * cell_w + (cols + 1) * gap)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        y = gap + r * (cell_h + gap)
        x = gap + c * (cell_w + gap)
        paste(out, border(fit(img, cell_h, cell_w)), y, x)
    return out


def navigation_rollout(factory, steps: int, size: int, seed: int = 0):
    from lumen.viz import render_frame

    env = factory()
    obs, _ = env.reset(seed=seed)
    frames = [render_frame(env, size=size)]
    infos = []
    for _ in range(steps):
        obs, _, terminated, truncated, info = env.step(np.array([1.0, 0.0], np.float32))
        infos.append(info)
        frames.append(render_frame(env, size=size))
        if terminated or truncated:
            break
    return frames, infos


def build_navigation(out: Path) -> dict:
    from lumen.envs.registration import make_nav_tortuous, make_tortuous_tree_nav

    tree_frames, tree_infos = navigation_rollout(make_tortuous_tree_nav, 100, 720)
    sten_frames, sten_infos = navigation_rollout(make_nav_tortuous, 80, 720)
    write_avi(out / "real-tree-navigation.avi", tree_frames, fps=12)
    write_avi(out / "real-tortuous-navigation.avi", sten_frames, fps=12)
    imgs = [
        tree_frames[0],
        tree_frames[len(tree_frames) // 2],
        tree_frames[-1],
        sten_frames[0],
        sten_frames[len(sten_frames) // 2],
        sten_frames[-1],
    ]
    write_png(out / "real-navigation-mosaic.png", montage(imgs, 2, 3, 330, 330))
    write_png(out / "real-tree-final.png", tree_frames[-1])
    write_png(out / "real-tortuous-final.png", sten_frames[-1])
    return {
        "tree_frames": len(tree_frames),
        "tree_final": tree_infos[-1] if tree_infos else {},
        "tortuous_frames": len(sten_frames),
        "tortuous_final": sten_infos[-1] if sten_infos else {},
    }


def build_modalities(out: Path) -> None:
    from lumen.assets import procedural
    from lumen.core.frame import CenterlineFrame
    from lumen.sensors import FluoroSensor, LuminalCamera, RealismParams

    asset = procedural.tortuous_tree(radius=4.0)
    trunk = np.asarray(asset.edges[0].centerline_mm, dtype=float)
    mid = np.asarray(asset.edges[1].centerline_mm, dtype=float)
    right = np.asarray(asset.edges[-1].centerline_mm, dtype=float)
    route = np.concatenate([trunk, mid[1:], right[1:]], axis=0)
    device = route[5:42:3]

    sensor = FluoroSensor(mu_device=1.25, res=96, n_samples=300, nu=256, nv=256)
    views = sensor.render_biplanar(
        device,
        radius=0.45,
        contrast_asset=asset,
        mu_contrast=0.11,
        contrast_eps=1.3,
        beer_lambert=False,
    )
    ap = np.flipud(views[0]["image"])
    lat = np.flipud(views[1]["image"])
    overlay = np.repeat((0.62 * _u8(ap))[:, :, None], 3, axis=2)
    vessel = np.flipud(views[0]["masks"]["vessel"])
    device_mask = np.flipud(views[0]["masks"]["device"])
    overlay[vessel, 1] = 190
    overlay[device_mask] = np.array([235, 76, 82], np.uint8)

    # Same route, different sensor: forward-looking luminal RGB near the inlet.
    pts, lumen = asset.edge_arrays(asset.edges[0])
    frame = CenterlineFrame(np.asarray(pts, float))
    scope_device = np.stack([pts[2], pts[7]])
    luminal = LuminalCamera(
        nu=256,
        nv=256,
        texture_strength=0.24,
        fold_strength=0.18,
        artifact_strength=0.22,
        artifact_seed=2,
    ).render(frame, lumen, scope_device)

    realism = RealismParams(i0=2500.0, psf_sigma=1.1, scatter_frac=0.13,
                            beam_hardening=0.05, read_noise=2.0, seed=4)
    noisy, _ = sensor.render(
        device,
        radius=0.45,
        contrast_asset=asset,
        mu_contrast=0.10,
        realism=realism,
        beer_lambert=False,
    )
    noisy = np.flipud(noisy)
    panels = [ap, lat, overlay, luminal, noisy, device_mask.astype(float)]
    write_png(out / "real-modalities-mosaic.png", montage(panels, 2, 3, 300, 300))
    write_png(out / "real-fluoro-ap.png", ap)
    write_png(out / "real-fluoro-lateral.png", lat)
    write_png(out / "real-fluoro-label-overlay.png", overlay)
    write_png(out / "real-luminal-rgb.png", luminal)
    write_png(out / "real-fluoro-noisy.png", noisy)


def build_advanced_metrics(out: Path) -> dict:
    from lumen.newton.aneurysm import Aneurysm, AneurysmSac
    from lumen.newton.clot import ClotField, ClotParams
    from lumen.newton.devices import FlowDiverter, Stentriever

    t = np.linspace(0, 14, 260)
    pressure = 100.0 + 42.0 * np.maximum(0.0, np.sin(2 * np.pi * 1.1 * t))
    aneurysm = Aneurysm(s_neck=45.0, neck_width=5.0, sac_volume=120.0,
                        wall_stiffness=2500.0, neck_resistance_coeff=16.0)
    sac_pre = AneurysmSac(aneurysm)
    sac_post = AneurysmSac(aneurysm)
    diverter = FlowDiverter(deployed_center=45.0, span=10.0, metal_coverage=0.35)
    div = diverter.diversion(aneurysm)
    q_pre, q_post = [], []
    dt = float(t[1] - t[0])
    for p in pressure:
        q_pre.append(abs(sac_pre.update(p, dt, diversion=0.0)))
        q_post.append(abs(sac_post.update(p, dt, diversion=div)))
    q_pre = np.asarray(q_pre)
    q_post = np.asarray(q_post)
    write_png(out / "real-aneurysm-flow-diversion.png", chart([
        (t, q_pre, AMBER),
        (t, q_post, CYAN),
    ]))

    clot = ClotField(80.0, 100, 16, 2.0, 35.0, 52.0, 1.5,
                     params=ClotParams(grip_coeff=0.24), n_envs=2)
    st = Stentriever(deployed_center=43.0, radial_force=0.2, n_struts=6)
    engagement = st.engagement_strength_for_mask(clot.s_grid, clot.mask_env[0])
    centers = []
    damage = []
    retrieved = []
    statuses = []
    steps = np.arange(10)
    for i in steps:
        asp = np.array([0.07, 0.0])
        result = clot.retrieve_batched(np.array([1.4, 1.4]), engagement, aspiration=asp)
        centers.append(np.nan_to_num(clot.clot_centers(), nan=0.0))
        damage.append(clot.D_env.max(axis=1))
        retrieved.append(clot.retrieved_env.copy())
        statuses.append([r["status"] for r in result])
    centers = np.asarray(centers)
    damage = np.asarray(damage)
    retrieved = np.asarray(retrieved)
    write_png(out / "real-stentriever-retrieval.png", chart([
        (steps, retrieved[:, 0], GREEN),
        (steps, retrieved[:, 1], RED),
        (steps, damage[:, 1] * max(float(retrieved[:, 0].max()), 1.0), VIOLET),
    ]))
    return {
        "flow_diverter_coverage": div,
        "flow_inflow_peak_without_diverter": float(q_pre.max()),
        "flow_inflow_peak_with_diverter": float(q_post.max()),
        "stentriever_statuses": statuses[-1],
        "stentriever_retrieved_env": retrieved[-1].tolist(),
        "stentriever_damage_env": damage[-1].tolist(),
    }


def ffmpeg_convert(src: Path, dst: Path, *, vf: str | None = None) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "navigation": build_navigation(out),
    }
    build_modalities(out)
    manifest["advanced_metrics"] = build_advanced_metrics(out)
    ffmpeg_convert(out / "real-tree-navigation.avi", out / "real-tree-navigation.mp4")
    ffmpeg_convert(out / "real-tortuous-navigation.avi", out / "real-tortuous-navigation.mp4")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="launch/captures/real")
    args = parser.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()
