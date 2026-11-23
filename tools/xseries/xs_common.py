"""Shared capture/render helpers for the Lumen X-series evidence clips.

Every frame is rendered from a real Lumen simulator state (Newton rod sim through
the public envs) and Lumen's own sensors (DRR fluoroscopy, masks, keypoints). Only
the 3-D panel uses matplotlib, drawing the same asset geometry and device nodes the
solver uses. Nothing here scripts device motion: positions come from `sim.body_positions()`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TAG = "Lumen (open source, Apache-2.0)"
FONT = os.environ.get("LUMEN_XS_FONT", "/System/Library/Fonts/Helvetica.ttc")


# --------------------------------------------------------------------------- output
def out_root() -> Path:
    root = Path(os.environ.get("LUMEN_XS_OUT", "/Volumes/LumenXS/runs/xseries"))
    real = root.resolve()
    if not str(real).startswith("/Volumes/") or str(real).startswith("/Volumes/Macintosh HD"):
        raise SystemExit(f"refusing to write outside an external volume: {real}")
    real.mkdir(parents=True, exist_ok=True)
    return real


def git_sha() -> str:
    here = Path(__file__).resolve().parents[2]
    return subprocess.check_output(["git", "-C", str(here), "rev-parse", "HEAD"], text=True).strip()


def git_dirty() -> bool:
    here = Path(__file__).resolve().parents[2]
    out = subprocess.check_output(["git", "-C", str(here), "status", "--porcelain", "--", "lumen"],
                                  text=True)
    return bool(out.strip())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_versions() -> dict:
    import importlib.metadata as md
    v = {"python": sys.version.split()[0], "platform": platform.platform(),
         "machine": platform.machine()}
    for d in ("warp-lang", "newton", "numpy", "matplotlib", "gymnasium", "stable-baselines3",
              "torch"):
        try:
            v[d] = md.version(d)
        except Exception:
            v[d] = None
    return v


# --------------------------------------------------------------------------- rollout
@dataclass
class Frame:
    nodes: np.ndarray
    info: dict
    wall_load: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def shaft_polyline(env, nodes):
    """Proximal wire behind the simulated distal rod.

    Lumen's TreeNavEnv simulates the distal 18 mm of the wire as a Newton rod (10 bodies,
    2 mm spacing); its base is actuated along the route centerline, i.e. the proximal shaft
    is a kinematic constraint that follows the route. For display we draw that shaft as the
    route centerline from the access point up to the rod base, then the rod itself."""
    f = getattr(env, "route_frame", None)
    if f is None:
        return np.asarray(nodes, float)
    nodes = np.asarray(nodes, float)
    s0 = float(f.project_s(nodes[:1])[0])
    keep = f.points[f.cum_s < s0 - 4.0]
    if len(keep) < 2:
        return nodes
    # The route polyline turns sharply at graph junctions; a real shaft cannot. Smooth the
    # displayed shaft (Chaikin corner cutting, ends fixed). Display only; rod untouched.
    raw = np.vstack([keep, nodes[:1]])
    seg = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    ss = np.unique(np.concatenate([np.arange(0.0, cs[-1], 8.0), [cs[-1]]]))
    coarse = np.stack([np.interp(ss, cs, raw[:, k]) for k in range(3)], axis=1)
    shaft = _chaikin(coarse, 5)
    return np.vstack([shaft[:-1], nodes])


def _chaikin(p, it=3):
    p = np.asarray(p, float)
    for _ in range(it):
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.empty((2 * len(q), 3))
        mid[0::2], mid[1::2] = q, r
        p = np.vstack([p[:1], mid, p[-1:]])
    return p


def rollout(env, policy, max_steps: int, seed: int = 0, record_wall=True, extra_fn=None):
    """Run one episode, recording solver node positions after reset and every step.
    `extra['wire']` holds shaft + rod (display polyline); `nodes` is the simulated rod."""
    env.reset(seed=seed)
    n0 = np.asarray(env.sim.body_positions(), float).copy()
    frames = [Frame(n0, {"step": 0}, None, {"wire": shaft_polyline(env, n0)})]
    obs = None
    for k in range(max_steps):
        a = policy(k, obs, env)
        obs, r, term, trunc, info = env.step(a)
        info = {kk: (float(vv) if isinstance(vv, (np.floating, float, int, np.integer))
                     and not isinstance(vv, bool) else vv) for kk, vv in info.items()}
        info["step"] = k + 1
        info["reward"] = float(r)
        wl = None
        if record_wall and hasattr(env.sim, "wall_load_grid"):
            wl = np.asarray(env.sim.wall_load_grid(), float).copy()
        ex = extra_fn(env) if extra_fn else {}
        nk = np.asarray(env.sim.body_positions(), float).copy()
        ex["wire"] = shaft_polyline(env, nk)
        frames.append(Frame(nk, info, wl, ex))
        if term or trunc:
            break
    return frames


def _rollout_code_version() -> str:
    """Short hash of the source that produces cached rollouts (Frame, rollout, shaft_polyline)."""
    import inspect
    mod = sys.modules[__name__]
    parts = []
    for nm in ("Frame", "rollout", "shaft_polyline"):
        obj = getattr(mod, nm, None)
        if obj is not None:
            try:
                parts.append(inspect.getsource(obj))
            except (OSError, TypeError):
                parts.append(nm)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:8]


def cached_rollout(name: str, make_env, policy, max_steps: int, seed: int = 0, extra_fn=None):
    """Run a rollout once and keep the solver states on the external drive, so clips that
    show the same episode (hero, auto-labels, safety, replay) use byte-identical states."""
    import pickle
    # The captured fields depend on extra_fn, so it is part of the cache identity: a cache
    # written without it is never served to a caller that needs it, and vice versa.
    tag = ""
    if extra_fn is not None:              # keyed on the function's source, not just its name
        import inspect
        try:
            src = inspect.getsource(extra_fn)
        except (OSError, TypeError):
            src = getattr(extra_fn, "__qualname__", "fn")
        tag = f"-x{getattr(extra_fn, '__name__', 'fn')}-{hashlib.sha256(src.encode()).hexdigest()[:8]}"
    # and on the code that records a rollout, so a cache written by older code is never reused
    pol = getattr(policy, "__qualname__", None) or type(policy).__name__
    pol = hashlib.sha256(pol.encode()).hexdigest()[:6]
    path = (out_root() / "rollouts" /
            f"{name}{tag}-v{_rollout_code_version()}-p{pol}-m{max_steps}-seed{seed}.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pickle.loads(path.read_bytes()), path
    env = make_env()
    t0 = time.time()
    frames = rollout(env, policy, max_steps, seed=seed, extra_fn=extra_fn)
    meta = {"sim_wall_s": round(time.time() - t0, 2),
            "safety_max_pen_mm": float(getattr(env, "safety_max_pen", float("nan"))),
            "dt_s_per_step": 5e-3 * float(getattr(env, "substeps", 4)),
            "device_radius_mm": float(getattr(env, "device_radius_mm", float("nan"))),
            "rod_bodies": int(len(frames[0].nodes))}
    path.write_bytes(pickle.dumps((frames, meta)))
    return (frames, meta), path


ARCH_TARGETS = {"lcca_out": "left common carotid", "bca_out": "brachiocephalic trunk",
                "lsa_out": "left subclavian"}


def arch_rollout(target="lcca_out", max_insertion=0.45, seed=0, max_steps=1500, extra_fn=None):
    """Constant-advance episode from the descending aorta into a supra-aortic branch of the
    femoral-approach arch (tools/xseries/anatomy.py)."""
    from lumen.envs.tree_nav import TreeNavEnv
    from tools.xseries.anatomy import femoral_arch
    asset = femoral_arch()

    def make():
        return TreeNavEnv(asset, target_node=target, max_steps=max_steps,
                          max_insertion=max_insertion)
    (frames, meta), path = cached_rollout(f"femarch-{target}-ins{max_insertion}", make,
                                          lambda k, o, e: np.array([1.0, 0.0]), max_steps,
                                          seed, extra_fn=extra_fn)
    tgt = np.array([n.position_mm for n in asset.nodes if n.id == target][0], float)
    return asset, tgt, frames, meta, path


# --------------------------------------------------------------------------- fluoro
class FluoroCam:
    """Fixed C-arm over the whole asset.

    Vessel: Lumen's DRR (voxelized contrast-filled lumen, ray-cast line integral), computed
    once because the lumen is static in the frame. Device: exact line integral of a cylinder
    of the device radius, projected through the same C-arm geometry (no voxel aliasing for a
    sub-millimetre wire). The sum goes through Lumen's detector-realism model
    (quantum noise, PSF, scatter, read noise), then Beer-Lambert to intensity.
    Display calibration is fixed per camera, never per frame."""

    def __init__(self, asset, axis=(0.0, 1.0, 0.0), up=None, nu=720, res=160, n_samples=128,
                 fov=1.3, margin=10.0, wire_attn=1.6, wire_min_px=1.1, realism=None,
                 contrast=True):
        from lumen.sensors.carm import CArm
        from lumen.sensors.drr import raycast
        from lumen.sensors.volume import asset_points, grid_for, voxelize_asset

        pts = asset_points(asset)
        self.grid = grid_for(pts, margin=margin, res=res)
        c = 0.5 * (pts.min(0) + pts.max(0))
        span = float(np.ptp(pts, axis=0).max()) + 2 * margin
        kw = {} if up is None else {"up": up}
        self.carm = CArm.looking_at(c, distance=2.0 * span, axis=axis, sdd=4.0 * span,
                                    width=fov * 2.0 * span, height=fov * 2.0 * span,
                                    nu=nu, nv=nu, **kw)
        self.n_samples, self.asset = n_samples, asset
        self.wire_attn, self.wire_min_px = wire_attn, wire_min_px
        if contrast:
            A_v = raycast(voxelize_asset(asset, self.grid, mu_device=1.0, eps=1.0),
                          self.grid, self.carm, n_samples=n_samples)
            # A_v is chord length through contrast (mm-equivalent up to a constant). Map to a
            # saturating roadmap opacity: 7 mm carotid ~0.75 of the 25 mm aorta, like iodinated
            # contrast at full concentration, where small vessels still read clearly.
            ref = max(1e-9, float(np.percentile(A_v[A_v > 0], 99.5)))
            L = 25.0 * A_v / ref
            self.A_vessel = 0.42 * (1.0 - np.exp(-L / 5.0))
        else:
            self.A_vessel = np.zeros((nu, nu))
        self.realism = realism
        self.params = {"axis": list(axis), "nu": nu, "vessel_volume_res": res,
                       "n_samples": n_samples, "fov_factor": fov, "magnification": 2.0,
                       "vessel_opacity": "0.42*(1-exp(-chord_mm/5)), DRR chord from Lumen raycast",
                       "device": "analytic cylinder line integral, attenuation %.1f at centre, "
                                 "min display half-width %.1f px" % (wire_attn, wire_min_px),
                       "detector": "lumen.sensors.realism.degrade i0=4e3 psf=1.1px scatter=0.08",
                       "display": "I = exp(-A), grey = 0.12 + 0.80 I"}

    def device_integral(self, nodes, radius=0.2):
        from lumen.sensors.fluoro import _projected_radius_px
        nu = self.carm.nu
        nodes = np.asarray(nodes, float)
        uv = np.array([self.carm.project(p) for p in nodes], float)
        rp = np.array([max(self.wire_min_px, _projected_radius_px(self.carm, p, radius))
                       for p in nodes])
        H, W = self.carm.nv, nu
        ss = 3                                           # 3x3 supersampling per pixel
        best = np.zeros((H * ss, W * ss))
        for a, b, ra, rb in zip(uv[:-1], uv[1:], rp[:-1], rp[1:]):
            if not np.isfinite([*a, *b]).all():
                continue
            m = max(ra, rb) + 1.0
            x0 = int(max(0, np.floor((min(a[0], b[0]) - m) * ss)))
            x1 = int(min(W * ss - 1, np.ceil((max(a[0], b[0]) + m) * ss)))
            y0 = int(max(0, np.floor((min(a[1], b[1]) - m) * ss)))
            y1 = int(min(H * ss - 1, np.ceil((max(a[1], b[1]) + m) * ss)))
            if x1 < x0 or y1 < y0:
                continue
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            X, Y = (xx + 0.5) / ss, (yy + 0.5) / ss      # supersample centres, pixel units
            ab = b - a
            den = float(ab @ ab)
            t = np.zeros_like(X) if den < 1e-12 else np.clip(
                ((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / den, 0, 1)
            dx, dy = X - (a[0] + t * ab[0]), Y - (a[1] + t * ab[1])
            r = ra + t * (rb - ra)
            q = np.sqrt(np.clip(1.0 - (dx * dx + dy * dy) / np.maximum(r * r, 1e-12), 0, None))
            win = best[y0:y1 + 1, x0:x1 + 1]
            np.maximum(win, q, out=win)                   # chord / diameter of a cylinder
        A = best.reshape(H, ss, W, ss).mean(axis=(1, 3))
        return self.wire_attn * A

    def line_integral(self, nodes, radius=0.2):
        return self.A_vessel + self.device_integral(nodes, radius)

    def image(self, nodes, radius=0.2, seed=0):
        from lumen.sensors.realism import RealismParams, degrade
        A = self.line_integral(nodes, radius)
        rp = self.realism or RealismParams(i0=4.0e3, psf_sigma=1.1, scatter_frac=0.08,
                                           scatter_sigma=24.0, read_noise=0.004, seed=seed)
        A = degrade(A, rp)
        I = np.exp(-np.clip(A, 0, None))             # Beer-Lambert; dense -> dark
        g = 0.12 + 0.80 * I                          # display window: background light grey
        return (np.clip(g, 0, 1) * 255).astype(np.uint8)

    def project(self, p):
        from lumen.sensors.fluoro import _keypoint
        return _keypoint(self.carm, np.asarray(p, float), (self.carm.nv, self.carm.nu))

    def device_mask(self, nodes, radius=0.2):
        from lumen.sensors.fluoro import _project_polyline_mask, _projected_radius_px
        shape = (self.carm.nv, self.carm.nu)
        dpx = [max(1.0, _projected_radius_px(self.carm, p, radius)) for p in nodes]
        return _project_polyline_mask(self.carm, nodes, shape, dpx)

    def vessel_mask(self):
        from lumen.sensors.fluoro import _project_asset_mask
        return _project_asset_mask(self.carm, self.asset, (self.carm.nv, self.carm.nu))

    def masks(self, nodes, radius=0.2):
        return {"device": self.device_mask(nodes, radius=radius),
                "vessel": self.vessel_mask()}


# --------------------------------------------------------------------------- 3-D panel
def _tube(pts, radii, n_theta=20):
    pts = np.asarray(pts, float)
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    ref = np.array([0.0, 1.0, 0.0])
    n1 = []
    prev = None
    for ti in t:
        r = ref if prev is None else prev
        v = r - ti * (r @ ti)
        if np.linalg.norm(v) < 1e-6:
            v = np.cross(ti, [1.0, 0.0, 0.0])
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            v = np.array([1.0, 0.0, 0.0])
        else:
            v = v / v_norm
        n1.append(v)
        prev = v
    n1 = np.asarray(n1)
    n2 = np.cross(t, n1)
    th = np.linspace(0, 2 * np.pi, n_theta)
    r = np.asarray(radii, float)[:, None]
    X = pts[:, None, :] + r[..., None] * (np.cos(th)[None, :, None] * n1[:, None, :]
                                         + np.sin(th)[None, :, None] * n2[:, None, :])
    return X[..., 0], X[..., 1], X[..., 2]


class View3D:
    def __init__(self, asset, size_px=(960, 1080), elev=12.0, azim0=-90.0, zoom=1.0,
                 wall_alpha=0.24):
        from lumen.sensors.volume import edge_radii
        self.asset, self.size, self.elev, self.azim0 = asset, size_px, elev, azim0
        self.tubes = []
        for e in asset.edges:
            p = np.asarray(e.centerline_mm, float)
            self.tubes.append(_tube(p, edge_radii(e, p)))
        allp = np.concatenate([np.asarray(e.centerline_mm, float) for e in asset.edges])
        self.c = allp.mean(0)
        self.h = float(np.ptp(allp, axis=0).max()) / 2 / zoom
        self.wall_alpha = wall_alpha

    def render(self, nodes, azim=None, target=None, tip_color="#5ce8ee", heat=None,
               device_color="#5ce8ee", title=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        w, h = self.size
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="#07090d")
        ax = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor="#07090d")
        for (X, Y, Z) in self.tubes:
            ax.plot_surface(X, Y, Z, color="#d4605c", alpha=self.wall_alpha, linewidth=0,
                            antialiased=True, shade=True)
        n = np.asarray(nodes, float)
        ax.plot(n[:, 0], n[:, 1], n[:, 2], color=device_color, lw=2.6, solid_capstyle="round")
        ax.scatter(*n[-1], s=36, color=tip_color, depthshade=False)
        if target is not None:
            ax.scatter(*target, s=110, facecolors="none", edgecolors="#f4cd54", linewidths=2,
                       depthshade=False)
        if heat is not None:
            # per-node wall indentation: cyan (none) -> amber (at threshold) -> red (over)
            pts, val, thr = heat
            pts, val = np.asarray(pts, float), np.asarray(val, float)
            u = np.clip(val / max(thr, 1e-9), 0.0, 1.0)[:, None]
            cyan, amber, red = (np.array([0.36, 0.91, 0.93]), np.array([0.96, 0.75, 0.30]),
                                np.array([0.95, 0.25, 0.25]))
            col = (1 - u) * cyan + u * amber
            col[val > thr] = red
            size = np.where(val > thr, 150.0, np.where(val > 0, 50.0, 14.0))
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#5ce8ee", lw=3.4)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col, s=size, depthshade=False,
                       alpha=1.0, edgecolors="none")
        c, hh = self.c, self.h
        ax.set_xlim(c[0] - hh, c[0] + hh); ax.set_ylim(c[1] - hh, c[1] + hh)
        ax.set_zlim(c[2] - hh, c[2] + hh)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=self.elev, azim=self.azim0 if azim is None else azim)
        ax.set_axis_off()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img


# --------------------------------------------------------------------------- compose
def font(size):
    """Helvetica on the authoring Mac; LUMEN_XS_FONT overrides; else DejaVu or PIL's default font."""
    from PIL import ImageFont
    for path in (FONT, "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)      # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def label(img, text, xy, size=26, color=(210, 218, 228), anchor="la", bg=None):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    f = font(size)
    if bg is not None:
        box = d.textbbox(xy, text, font=f, anchor=anchor)
        d.rectangle((box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6), fill=bg)
    d.text(xy, text, font=f, fill=color, anchor=anchor)


def corner_tag(img, top=False):
    W, H = img.size
    xy, anchor = ((W - 22, 14), "ra") if top else ((W - 22, H - 20), "rd")
    label(img, TAG, xy, size=max(18, H // 46), color=(225, 230, 236), anchor=anchor, bg=(10, 13, 18))


def gray_to_rgb(g):
    return np.repeat(np.asarray(g)[..., None], 3, axis=2)


def encode(frame_dir: Path, out_mp4: Path, fps=30) -> list[str]:
    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i",
           str(frame_dir / "%05d.png"), "-an", "-c:v", "libx264", "-preset", "slow", "-crf",
           "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)]
    subprocess.run(cmd, check=True)
    return cmd


def probe(mp4: Path) -> dict:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                   "-show_entries", "stream=width,height,nb_frames,r_frame_rate:format=duration",
                                   "-of", "json", str(mp4)], text=True)
    j = json.loads(out)
    s = j["streams"][0]
    return {"width": s["width"], "height": s["height"], "frames": int(s.get("nb_frames", 0)),
            "duration_s": round(float(j["format"]["duration"]), 2),
            "audio_streams": 0}


def write_manifest(path: Path, record: dict):
    record = dict(record)
    record.setdefault("created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path.write_text(json.dumps(record, indent=2, default=lambda o: o.tolist()
                               if hasattr(o, "tolist") else str(o)))
