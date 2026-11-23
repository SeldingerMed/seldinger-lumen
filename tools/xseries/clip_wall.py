"""Clip 4: deformable vessel wall. The same fast push (0.9 mm/step, 45 mm/s) from the
descending aorta into the left common carotid, run twice in the same procedural arch:
once against Lumen's default rigid wall and once with the HGO (Holzapfel-Gasser-Ogden)
anisotropic hyperelastic wall switched on (`deformable_wall=True`, library-default HGO
parameters). The deformable wall is drawn at true scale from the solver's own radial
deflection field w(s, theta): every wall vertex sits at R0 + w in the same edge frame and
angular bins the contact kernel uses. Colour shows the local deflection.

    python -m tools.xseries.clip_wall
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.xseries.xs_common import (arch_rollout, cached_rollout, corner_tag, encode, env_versions,
                                     font, git_dirty, git_sha, label, out_root, probe, sha256,
                                     write_manifest)

LZ = Image.Resampling.LANCZOS
DT = 0.02
INS = 0.9
TARGET = "lcca_out"


def deformable_env_cls():
    from lumen.envs.tree_nav import TreeNavEnv
    from lumen.newton.sim import NewtonGuidewireSim

    class DeformableTreeNav(TreeNavEnv):
        """TreeNavEnv with the solver's HGO wall enabled (library-default HGOParams)."""

        def reset(self, *, seed=None, options=None):
            if getattr(self, "sim", None) is None:
                self.sim = NewtonGuidewireSim(
                    np.asarray(self.route_frame.points), self.R, self._device_points(),
                    radius=self.device_radius_mm, kappa=3e3, d_hat=0.3, vbd_iterations=8,
                    device=self.device, tree=self.tree, deformable_wall=True,
                    route_centerline=np.asarray(self.route_frame.points))
            return super().reset(seed=seed, options=options)
    return DeformableTreeNav


def deformable_rollout(seed=0, max_steps=1500):
    from tools.xseries.anatomy import femoral_arch
    asset = femoral_arch()
    cls = deformable_env_cls()

    def make():
        return cls(asset, target_node=TARGET, max_steps=max_steps, max_insertion=INS)

    def extra(env):
        s = env.sim.solver
        W = s._tree_wall
        w = W.w_field.numpy().reshape(s._tree_n_edges, W.n_s, W.n_th).astype(np.float32).copy()
        return {"w": w, "pen": np.asarray(env.sim.surface_penetration(), float)}

    (frames, meta), path = cached_rollout(f"femarch-{TARGET}-ins{INS}-hgo", make,
                                          lambda k, o, e: np.array([1.0, 0.0]), max_steps, seed,
                                          extra_fn=extra)
    return asset, frames, meta, path


class WallView:
    """3-D view of the vascular tree with each edge's wall drawn from w(s, theta)."""

    def __init__(self, asset, size_px=(960, 760), elev=10.0, azim=-90.0, zoom=1.9, dz=22.0):
        from lumen.envs.tree_nav import TreeNavEnv
        from lumen.newton.hgo_wall import HGOParams
        self.tree = TreeNavEnv(asset, target_node=TARGET).tree
        self.size, self.elev, self.azim = size_px, elev, azim
        self.hgo = HGOParams()
        self.geom = []
        for e in self.tree.edges:
            P = np.asarray(e.frame.points, float)
            T = np.asarray(e.frame.tangents, float)
            M1 = np.asarray(e.frame.m1, float)
            M1 = M1 - (M1 * T).sum(1, keepdims=True) * T
            M1 /= np.linalg.norm(M1, axis=1, keepdims=True) + 1e-12
            M2 = np.cross(T, M1)
            R = np.asarray(e.lf.R, float).reshape(len(P), -1).mean(1)
            self.geom.append((P, M1, M2, R, np.asarray(e.frame.cum_s, float), float(e.frame.length)))
        allp = np.concatenate([g[0] for g in self.geom])
        self.c = allp.mean(0) + np.array([0.0, 0.0, dz])
        self.h = float(np.ptp(allp, axis=0).max()) / 2 / zoom

    def surfaces(self, w=None, n_th=16):
        out = []
        th_c = -np.pi + 2 * np.pi * (np.arange(n_th) + 0.5) / n_th     # kernel bin centres
        th = np.concatenate([th_c, th_c[:1] + 2 * np.pi])
        for ei, (P, M1, M2, R, cs, L) in enumerate(self.geom):
            if w is None:
                ww = np.zeros((len(P), n_th))
            else:
                n_s = w.shape[1]
                si = np.clip(np.rint(cs / L * (n_s - 1)).astype(int), 0, n_s - 1)  # kernel s binning
                ww = w[ei][si]
            ww = np.concatenate([ww, ww[:, :1]], axis=1)
            r = R[:, None] + ww
            nrm = (np.cos(th)[None, :, None] * M1[:, None, :] + np.sin(th)[None, :, None] * M2[:, None, :])
            X = P[:, None, :] + r[..., None] * nrm
            out.append((X, ww, nrm))
        return out

    def light(self, nrm):
        """Lambert + rim shading from the camera direction (facecolors bypass mpl shading)."""
        el, az = np.radians(self.elev), np.radians(self.azim)
        cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        key = cam + np.array([-0.4, 0.0, 0.6])
        key /= np.linalg.norm(key)
        lam = np.clip((nrm * key).sum(-1), 0, 1)
        return 0.45 + 0.55 * lam

    def render(self, wire, w=None, target=None, heat=None, vmax=0.5):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        W, H = self.size
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="#07090d")
        ax = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor="#07090d")
        cmap = colormaps["inferno"]
        base = np.array([0.83, 0.38, 0.36, 0.22])
        for X, ww, nrm in self.surfaces(w):
            u = np.clip(ww / vmax, 0, 1)
            fc = np.where((u > 0.01)[..., None],
                          np.concatenate([cmap(0.35 + 0.65 * u)[..., :3], (0.35 + 0.6 * u)[..., None]], -1),
                          np.broadcast_to(base, u.shape + (4,))).copy()
            fc[..., :3] *= self.light(nrm)[..., None]
            ax.plot_surface(X[..., 0], X[..., 1], X[..., 2], facecolors=fc[:-1, :-1], linewidth=0,
                            antialiased=True, shade=False)
        n = np.asarray(wire, float)
        ax.plot(n[:, 0], n[:, 1], n[:, 2], color="#5ce8ee", lw=2.6, solid_capstyle="round")
        ax.scatter(*n[-1], s=36, color="#5ce8ee", depthshade=False)
        if heat is not None:
            pts, val, thr = heat
            pts, val = np.asarray(pts, float), np.asarray(val, float)
            over = val > thr
            if over.any():
                ax.scatter(pts[over, 0], pts[over, 1], pts[over, 2], c="#f04040", s=150,
                           depthshade=False, edgecolors="none")
        if target is not None:
            ax.scatter(*target, s=110, facecolors="none", edgecolors="#f4cd54", linewidths=2,
                       depthshade=False)
        c, hh = self.c, self.h
        ax.set_xlim(c[0] - hh, c[0] + hh); ax.set_ylim(c[1] - hh, c[1] + hh)
        ax.set_zlim(c[2] - hh, c[2] + hh)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.set_axis_off()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img


def _legend(img, xy, vmax):
    from matplotlib import colormaps
    cmap = colormaps["inferno"]
    d = ImageDraw.Draw(img)
    x0, y0 = xy
    for k in range(200):
        c = tuple(int(255 * v) for v in cmap(0.35 + 0.65 * k / 199)[:3])
        d.line((x0 + k, y0, x0 + k, y0 + 12), fill=c)
    d.text((x0, y0 + 18), "0", font=font(15), fill=(160, 168, 180), anchor="la")
    d.text((x0 + 200, y0 + 18), f"{vmax:.1f}+ mm", font=font(15), fill=(160, 168, 180), anchor="ra")
    d.text((x0, y0 - 6), "wall deflection", font=font(15), fill=(160, 168, 180), anchor="ld")


def chart(w, h, t_now, series, tmax, ymax, thr):
    img = Image.new("RGB", (w, h), (11, 14, 19))
    d = ImageDraw.Draw(img)
    L, R, T, B = 70, 20, 34, 40
    pw, ph = w - L - R, h - T - B
    d.rectangle((L, T, L + pw, T + ph), outline=(52, 62, 76))
    y_thr = T + ph * (1 - thr / ymax)
    d.line((L, y_thr, L + pw, y_thr), fill=(236, 185, 88), width=1)
    d.text((L + pw - 6, y_thr - 4), f"{thr:.1f} mm overlap threshold", font=font(15),
           fill=(236, 185, 88), anchor="rd")
    step = 0.5 if ymax > 1.5 else 0.2
    for yv in np.arange(0, ymax + 1e-9, step):
        yy = T + ph * (1 - yv / ymax)
        d.text((L - 8, yy), f"{yv:.1f}", font=font(15), fill=(130, 140, 152), anchor="rm")
    d.text((8, T - 26), "mm", font=font(17), fill=(190, 198, 208))
    x_leg = 60
    for name, _, color in series:
        d.line((x_leg, T - 16, x_leg + 26, T - 16), fill=color, width=4)
        d.text((x_leg + 34, T - 16), name, font=font(17), fill=color, anchor="lm")
        x_leg += 40 + int(d.textlength(name, font=font(17)))
    d.text((L, T + ph + 22), f"simulated time   t = {t_now:4.2f} s", font=font(16),
           fill=(130, 140, 152), anchor="lm")
    for _, vals, color in series:
        pts = [(L + pw * (k * DT) / tmax, T + ph * (1 - min(v, ymax) / ymax))
               for k, v in enumerate(vals) if k * DT <= t_now + 1e-9]
        if len(pts) > 1:
            d.line(pts, fill=color, width=3)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold", type=int, default=60)
    ap.add_argument("--only", default="", help="comma frame indices: preview stills, no encode")
    a = ap.parse_args(argv)
    only = [int(x) for x in a.only.split(",") if x]
    out = Path(a.out) if a.out else out_root() / "clip4-deformable-wall"
    out.mkdir(parents=True, exist_ok=True)

    def pen_extra(env):
        return {"pen": np.asarray(env.sim.surface_penetration(), float)}
    asset, target, rigid, rmeta, rpath = arch_rollout(TARGET, INS, seed=a.seed, extra_fn=pen_extra)
    _, soft, smeta, spath = deformable_rollout(a.seed)
    thr = float(rmeta["safety_max_pen_mm"])
    wmax = [float(f.extra["w"].max()) if "w" in f.extra else 0.0 for f in soft]
    # Effective overlap against the DEFORMED wall: base-geometry overlap minus local deflection
    # is not separable per node here, so report both raw quantities and say which is which.
    rpen = [f.info.get("max_pen", 0.0) for f in rigid]
    spen = [f.info.get("max_pen", 0.0) for f in soft]
    summ = {"rigid": {"steps": len(rigid) - 1, "success": rigid[-1].info.get("success"),
                      "peak_overlap_mm": round(max(rpen), 4),
                      "peak_wall_load_sim_units": round(max(f.info.get("wall_load_max", 0.0) for f in rigid[1:]), 1)},
            "deformable": {"steps": len(soft) - 1, "success": soft[-1].info.get("success"),
                           "peak_overlap_vs_undeformed_wall_mm": round(max(spen), 4),
                           "peak_wall_deflection_mm": round(max(wmax), 4),
                           "steps_with_deflection_over_0.01mm": int(sum(x > 0.01 for x in wmax)),
                           "peak_wall_load_sim_units": round(max(f.info.get("wall_load_max", 0.0) for f in soft[1:]), 1)}}
    print(summ, flush=True)

    rv = WallView(asset)
    nst = max(len(rigid), len(soft))
    rp = rpen + [rpen[-1]] * (nst - len(rpen))
    wm = wmax + [wmax[-1]] * (nst - len(wmax))
    # slow motion (x4) on steps where either wall is in contact, so the events are watchable
    sched = []
    for k in range(nst):
        slow = wm[k] > 0.02 or rp[k] > 0.05
        sched += [(k, slow)] * (4 if slow else 1)
    sched += [(nst - 1, False)] * a.hold
    n_frames = len(sched)
    dt = float(rmeta.get("dt_s_per_step", DT))
    tmax = (max(len(rigid), len(soft)) - 1) * dt
    vmax = 0.5
    ymax = max(1.0, 1.1 * max(max(rpen), min(max(wmax), 4.0)))
    red, amber = (232, 90, 92), (247, 181, 64)
    dirs = {k: out / f"frames_{k}" for k in ("landscape", "square")}
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for i in range(n_frames):
        if only and i not in only:
            continue
        k, slow = sched[i]
        fr = rigid[min(k, len(rigid) - 1)]
        fs = soft[min(k, len(soft) - 1)]
        t_now = k * dt
        pr = Image.fromarray(rv.render(fr.extra["wire"], None, target, heat=(fr.nodes, fr.extra.get("pen", np.zeros(len(fr.nodes))), thr)))
        ps = Image.fromarray(rv.render(fs.extra["wire"], fs.extra.get("w"), target, vmax=vmax))
        wnow = float(fs.extra["w"].max()) if "w" in fs.extra else 0.0
        wpk = max(wmax[:min(k, len(soft) - 1) + 1])
        L = Image.new("RGB", (1920, 1080), (7, 9, 13))
        L.paste(pr, (0, 0)); L.paste(ps, (960, 0))
        label(L, "Rigid wall", (24, 18), 28, red)
        label(L, f"device-wall overlap {fr.info.get('max_pen', 0.0):.2f} mm", (24, 58), 20, (190, 198, 208))
        label(L, "Deformable HGO wall", (984, 18), 28, amber)
        label(L, f"wall deflection now {wnow:.2f} mm   peak {wpk:.2f} mm   (true scale)", (984, 58), 20,
              (190, 198, 208))
        _legend(L, (1650, 640), vmax)
        if slow:
            label(L, "slow motion x4", (960, 96), 20, (236, 185, 88), anchor="ma", bg=(10, 13, 18))
        label(L, "Same push: 45 mm/s into the left common carotid", (960, 712), 22, (170, 178, 190), anchor="ma",
              bg=(10, 13, 18))
        L.paste(chart(1920, 320, t_now, [("rigid: device-wall overlap", rpen, red),
                                         ("deformable: peak wall deflection", wmax, amber)],
                      tmax, ymax, thr, dt=dt), (0, 760))
        corner_tag(L, top=True)
        L.save(dirs["landscape"] / f"{i:05d}.png")
        S = Image.new("RGB", (1080, 1080), (7, 9, 13))
        S.paste(pr.crop((150, 40, 810, 700)).resize((540, 540), LZ), (0, 40))
        S.paste(ps.crop((150, 40, 810, 700)).resize((540, 540), LZ), (540, 40))
        label(S, "Rigid wall", (16, 8), 24, red); label(S, "Deformable HGO wall", (556, 8), 24, amber)
        label(S, f"overlap {fr.info.get('max_pen', 0.0):.2f} mm", (16, 548), 20, (190, 198, 208))
        label(S, f"deflection {wnow:.2f} mm", (556, 548), 20, (190, 198, 208))
        if slow:
            label(S, "slow motion x4", (540, 52), 20, (236, 185, 88), anchor="ma", bg=(10, 13, 18))
        S.paste(chart(1080, 440, t_now, [("rigid overlap", rpen, red), ("wall deflection", wmax, amber)],
                      tmax, ymax, thr, dt=dt), (0, 600))
        corner_tag(S)
        S.save(dirs["square"] / f"{i:05d}.png")
        if i % 40 == 0:
            print(f"frame {i}/{n_frames}", flush=True)
    if only:
        print("preview written")
        return
    from lumen.newton.hgo_wall import HGOParams
    rec = {"clip": "4-deformable-wall",
           "description": "The same constant-speed push (0.9 mm/step = 45 mm/s) from the descending aorta into "
           "the left common carotid of the procedural femoral-approach arch, run against Lumen's rigid wall "
           "(left) and with the HGO anisotropic hyperelastic wall enabled (right, deformable_wall=True, "
           "library-default HGOParams). The deformable wall is drawn at true scale from the solver's radial "
           "deflection field w(s, theta), using the same edge frames and angular/arc-length bins as the "
           "contact kernel; colour encodes local deflection (0 to 0.5 mm, inferno). Left: device nodes over "
           "the 0.3 mm overlap threshold in red. Wall load is in simulator units (no SI calibration in the "
           "open core). Steps where either wall is in contact play at quarter speed (labelled 'slow motion x4').",
           "hgo_params": vars(HGOParams()), "summary": summ, "commit": git_sha(),
           "lumen_tree_dirty": git_dirty(),
           "command": "python -m tools.xseries.clip_wall " + " ".join(sys.argv[1:] if argv is None else argv),
           "rollout_files": [str(rpath), str(spath)], "sim": smeta, "versions": env_versions(),
           "outputs": {}}
    for k, d in dirs.items():
        mp4 = out / f"clip4-deformable-wall-{k}.mp4"
        rec["encode_cmd"] = encode(d, mp4)
        rec["outputs"][k] = {"file": mp4.name, "sha256": sha256(mp4), **probe(mp4)}
    write_manifest(out / "manifest.json", rec)
    print(rec["outputs"])


if __name__ == "__main__":
    main()
