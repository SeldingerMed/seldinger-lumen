"""Clip 8: Lumen vs CathSim, matched evaluation episodes, every training seed.

Left: Lumen NavTreeBranch (bifurcation, target branch). Right: CathSim phantom3, target BCA.
Each tile is one training seed's frozen PPO policy (600k steps, preregistered main protocol)
playing the first common evaluation seed (10000) deterministically. Episodes come from
benchmarks/external_comparison/record_matched.py; nothing is re-simulated here.

Both sides use the same drawing: the vessel lumen as a projected density (sum of surface
samples along the view axis, like a contrast projection), the wire polyline, the target ring,
and a per-tile readout of that simulator's own safety quantity against its stated threshold
(Lumen: wall overlap, mm, limit 0.3 mm; CathSim: tip-wall contact force, N, limit 1.0 N).
Playback is synchronized on simulated time, slowed 2x. Tiles freeze when their episode ends.

    python -m tools.xseries.clip_matched --episodes DIR [--compare compare.json]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.xseries.xs_common import (corner_tag, encode, env_versions, git_dirty, git_sha, label, out_root,
                                     probe, sha256, write_manifest)

SLOW = 2.0
FPS = 30
LIMIT = {"lumen": 0.3, "cathsim": 1.0}
UNIT = {"lumen": "mm overlap", "cathsim": "N tip force"}
TITLE = {"lumen": "Lumen  |  bifurcation, target branch", "cathsim": "CathSim  |  aortic arch phantom, target BCA"}
WIRE = (110, 240, 245)


def view_basis(V, env, tgt):
    """Frontal projection plane = the vessel's two principal axes; screen-up points from the
    vessel centroid toward the target, so the target branch is drawn upward on both sides."""
    if env == "lumen":
        return np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    C = V - V.mean(0)
    _, _, Vt = np.linalg.svd(C[:: max(1, len(C) // 20000)], full_matrices=False)
    n = Vt[2]
    up = np.asarray(tgt, float) - V.mean(0)
    up = up - (up @ n) * n
    up /= np.linalg.norm(up)
    right = np.cross(up, n)
    return right, up


class Tile:
    def __init__(self, d, w, h):
        self.env = str(d["environment"])
        self.V = np.asarray(d["vessel"], float)
        self.e1, self.e2 = view_basis(self.V, self.env, d["target"])
        self.wire = list(d["wire"])
        self.safety = np.asarray(d["safety"], float)
        self.dt = float(d["dt"])
        self.steps = int(d["steps"])
        self.success = bool(d["success"])
        self.seed = int(d["training_seed"])
        self.tgt = np.asarray(d["target"], float)
        self.w, self.h = w, h
        P = self.p2(self.V)
        lo, hi = P.min(0), P.max(0)
        pad = 14
        self.s = min((w - 2 * pad) / (hi - lo)[0], (h - 2 * pad - 40) / (hi - lo)[1])
        self.off = np.array([(w - (hi - lo)[0] * self.s) / 2, 40 + (h - 40 - (hi - lo)[1] * self.s) / 2])
        self.lo, self.hi = lo, hi
        self.bg = self.background()

    def p2(self, X):
        X = np.atleast_2d(X)
        return np.stack([X @ self.e1, X @ self.e2], 1)

    def uv(self, X):
        P = self.p2(X)
        u = self.off[0] + (P[:, 0] - self.lo[0]) * self.s
        v = self.h - self.off[1] + 40 - (P[:, 1] - self.lo[1]) * self.s
        return np.stack([u, v], 1)

    def background(self):
        uv = self.uv(self.V)
        H, _, _ = np.histogram2d(uv[:, 1], uv[:, 0], bins=[self.h, self.w], range=[[0, self.h], [0, self.w]])
        k = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1], float)
        k /= k.sum()
        for ax in (0, 1):
            H = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), ax, H)
        H = H / (np.percentile(H[H > 0], 97) + 1e-9)
        H = np.clip(H, 0, 1) ** 0.6
        base = np.array([10, 13, 18], float)
        ves = np.array([196, 92, 88], float)
        img = base[None, None, :] * (1 - H[..., None]) + ves[None, None, :] * H[..., None]
        return Image.fromarray(img.astype(np.uint8))

    def frame(self, t):
        k = min(self.steps, int(t / self.dt))
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        tg = self.uv(self.tgt)[0]
        d.ellipse((tg[0] - 9, tg[1] - 9, tg[0] + 9, tg[1] + 9), outline=(112, 211, 141), width=2)
        w = np.asarray(self.wire[k], float)
        # CathSim feeds the wire from a base outside the phantom; draw only the part within the
        # vessel's extent (3 mm margin) so the external feed segment does not cross the image.
        inside = np.all((w >= self.V.min(0) - 3.0) & (w <= self.V.max(0) + 3.0), axis=1)
        if inside.any():
            w = w[int(np.argmax(inside)):]
        if len(w) > 1:
            d.line([tuple(x) for x in self.uv(w)], fill=WIRE, width=3, joint="curve")
        tp = self.uv(w[-1])[0]
        d.ellipse((tp[0] - 4, tp[1] - 4, tp[0] + 4, tp[1] + 4), fill=(240, 250, 252))
        peak = float(self.safety[: k + 1].max())
        over = peak > LIMIT[self.env]
        label(img, f"training seed {self.seed}", (10, 8), 18, (200, 206, 214))
        col = (236, 110, 96) if over else (160, 168, 180)
        label(img, f"peak {peak:.2f} {UNIT[self.env]}", (10, 32), 16, col)
        done = k >= self.steps
        if done:
            ok = self.success and not over
            txt = "reached, safe" if ok else ("reached, over limit" if self.success else "did not reach")
            label(img, txt, (self.w - 10, 8), 18, (112, 211, 141) if ok else (236, 185, 88), anchor="ra")
        else:
            label(img, f"{k * self.dt:4.2f} s", (self.w - 10, 8), 18, (200, 206, 214), anchor="ra")
        return img


def load(ep_dir, env, seeds=range(6)):
    out = []
    for s in seeds:
        f = Path(ep_dir) / f"{env}-s{s}.npz"
        if f.exists():
            out.append(np.load(f, allow_pickle=True))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    only = [int(x) for x in a.only.split(",") if x]
    out = Path(a.out) if a.out else out_root() / "clip8-lumen-vs-cathsim"
    out.mkdir(parents=True, exist_ok=True)
    cmp = json.load(open(a.compare)) if a.compare and Path(a.compare).exists() else None
    seeds = (cmp or {}).get("training_seeds") or range(6)
    L, C = load(a.episodes, "lumen", seeds), load(a.episodes, "cathsim", seeds)
    layouts = {"landscape": ((1920, 1080), (300, 440), [(20, 150), (980, 150)], (3, 2)),
               "square": ((1080, 1080), (340, 205), [(30, 92), (30, 590)], (3, 2))}
    T = max([x["steps"] * float(x["dt"]) for x in L + C] + [0.1])
    n_frames = int(T * SLOW * FPS) + int(a.hold * FPS)
    rec = {"clip": "8-lumen-vs-cathsim", "commit": git_sha(), "lumen_tree_dirty": git_dirty(),
           "command": "python -m tools.xseries.clip_matched " + " ".join(sys.argv[1:] if argv is None else argv),
           "versions": env_versions(), "episodes": {}, "outputs": {}}
    for d in L + C:
        rec["episodes"][f"{d['environment']}-s{int(d['training_seed'])}"] = {
            "success": bool(d["success"]), "steps": int(d["steps"]), "dt_s": float(d["dt"]),
            "peak_safety": float(np.max(d["safety"])), "eval_seed": int(d["eval_seed"]), "model": str(d["model"])}
    for kind, (size, (tw, th), origins, (cols, rows)) in layouts.items():
        fd = out / f"frames_{kind}"
        shutil.rmtree(fd, ignore_errors=True)
        fd.mkdir(parents=True)
        sides = [[Tile(d, tw, th) for d in L], [Tile(d, tw, th) for d in C]]
        for i in range(n_frames):
            if only and i not in only:
                continue
            t = i / (FPS * SLOW)
            img = Image.new("RGB", size, (7, 9, 13))
            for side, (ox, oy), env in zip(sides, origins, ("lumen", "cathsim")):
                label(img, TITLE[env], (ox, oy - (40 if kind == "landscape" else 34)), 24 if kind == "landscape"
                      else 20, (235, 240, 245))
                for j, tile in enumerate(side):
                    gap = 10
                    x = ox + (j % cols) * (tw + gap)
                    y = oy + (j // cols) * (th + gap)
                    img.paste(tile.frame(t), (x, y))
                if not side:
                    label(img, "episodes pending", (ox + 20, oy + 40), 22, (236, 185, 88))
            if kind == "landscape":
                label(img, "Lumen vs CathSim: matched PPO policies (600k env steps each), every training seed, "
                           "same evaluation seed", (20, 24), 30)
                label(img, f"Safety limits: Lumen wall overlap 0.3 mm (contact barrier width); CathSim tip-wall "
                           f"force 1.0 N.   Simulated time, slowed {SLOW:.0f}x.", (20, 66), 20, (160, 168, 180))
            else:
                label(img, "Lumen vs CathSim: matched PPO, every training seed", (30, 16), 26)
                label(img, f"limits: Lumen 0.3 mm overlap, CathSim 1.0 N tip force; {SLOW:.0f}x slow", (30, 50), 18,
                      (160, 168, 180))
            if cmp and i >= n_frames - int(a.hold * FPS):
                Ls, Cs = cmp["lumen"]["summary"], cmp["cathsim"]["summary"]
                if Cs["success_rate"]["mean"] is not None:
                    txt = (f"Mean over {len(cmp.get('training_seeds') or range(6))} training seeds x 100 eval seeds:  success  Lumen "
                           f"{100 * Ls['success_rate']['mean']:.0f}%  CathSim {100 * Cs['success_rate']['mean']:.0f}%"
                           f"   |   safe success  Lumen {100 * Ls['safe_success_rate']['mean']:.0f}%  CathSim "
                           f"{100 * Cs['safe_success_rate']['mean']:.0f}%")
                    y = 1080 - 60 if kind == "landscape" else 1080 - 40
                    label(img, txt, (size[0] // 2, y), 22 if kind == "landscape" else 15, (235, 240, 245),
                          anchor="ma", bg=(10, 13, 18))
            corner_tag(img, top=(kind == "landscape"))
            img.save(fd / f"{i:05d}.png")
        if only:
            continue
        mp4 = out / f"clip8-lumen-vs-cathsim-{kind}.mp4"
        rec["encode_cmd"] = encode(fd, mp4, fps=FPS)
        rec["outputs"][kind] = {"file": mp4.name, "sha256": sha256(mp4), **probe(mp4)}
    if only:
        print("preview written")
        return
    write_manifest(out / "manifest.json", rec)
    print(rec["outputs"])


if __name__ == "__main__":
    main()
