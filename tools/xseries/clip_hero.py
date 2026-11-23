"""Clip 1 (hero): guidewire advanced from the ascending aorta through the arch into
the left common carotid. Left: Lumen synthetic fluoroscopy (DRR + detector model).
Right: the same solver state in 3-D. Every frame is one solver step.

    python -m tools.xseries.clip_hero
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tools.xseries.xs_common import (ARCH_TARGETS, FluoroCam, View3D, arch_rollout, corner_tag,
                                     encode, env_versions, git_dirty, git_sha, gray_to_rgb, label,
                                     out_root, probe, sha256, write_manifest)

LZ = Image.Resampling.LANCZOS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nu", type=int, default=720)
    ap.add_argument("--target", default="lcca_out", choices=sorted(ARCH_TARGETS))
    ap.add_argument("--insertion", type=float, default=0.45, help="mm per env step")
    ap.add_argument("--hold", type=int, default=50, help="frames to hold the final state")
    ap.add_argument("--only", default="", help="comma frame indices: preview stills, no encode")
    a = ap.parse_args(argv)

    out = Path(a.out) if a.out else out_root() / "clip1-hero"
    out.mkdir(parents=True, exist_ok=True)
    asset, target, frames, meta, rpath = arch_rollout(a.target, a.insertion, seed=a.seed)
    vname = ARCH_TARGETS[a.target]
    only = [int(x) for x in a.only.split(",") if x]
    last = frames[-1].info
    pens = [f.info.get("max_pen", 0.0) for f in frames[1:]]

    cam = FluoroCam(asset, axis=(0.0, 1.0, 0.0), nu=a.nu, fov=1.0)
    view = View3D(asset, size_px=(960, 1080), elev=14.0, zoom=1.45)
    view_sq = View3D(asset, size_px=(540, 1000), elev=14.0, zoom=1.75)
    dirs = {k: out / f"frames_{k}" for k in ("landscape", "square")}
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    seq = list(range(len(frames))) + [len(frames) - 1] * a.hold
    n = len(seq)
    for i, fi in enumerate(seq):
        if only and i not in only:
            continue
        fr = frames[fi]
        wire = fr.extra["wire"]
        g = np.flipud(cam.image(wire, seed=fi))
        fl = Image.fromarray(gray_to_rgb(g)).resize((960, 960), LZ)
        az = -90.0 + 50.0 * np.sin(np.pi * i / max(1, n - 1))           # slow orbit
        v3 = Image.fromarray(view.render(wire, azim=az, target=target))
        L = Image.new("RGB", (1920, 1080), (7, 9, 13))
        L.paste(fl, (0, 60)); L.paste(v3, (960, 0))
        label(L, "Synthetic fluoroscopy", (24, 18), 26)
        label(L, "Solver state, 3-D", (984, 18), 26)
        label(L, f"Descending aorta to {vname}    t = {fr.info['step'] * meta['dt_s_per_step']:4.1f} s",
              (984, 1040), 22, (150, 160, 172))
        corner_tag(L)
        L.save(dirs["landscape"] / f"{i:05d}.png")
        S = Image.new("RGB", (1080, 1080), (7, 9, 13))
        S.paste(fl.crop((240, 0, 720, 960)).resize((500, 1000), LZ), (20, 70))
        S.paste(Image.fromarray(view_sq.render(wire, azim=az, target=target)), (540, 70))
        label(S, "Synthetic fluoroscopy", (18, 22), 24)
        label(S, "Solver state, 3-D", (558, 22), 24)
        corner_tag(S)
        S.save(dirs["square"] / f"{i:05d}.png")
        if i % 50 == 0:
            print(f"frame {i}/{n}", flush=True)
    if only:
        print("preview written", dirs)
        return

    rec = {"clip": "1-hero",
           "description": f"Guidewire advanced from the descending thoracic aorta across a "
           f"procedural, adult-proportioned aortic arch into the {vname}. The distal 18 mm is "
           "Lumen's Newton rod (Cosserat rod, VBD solver, barrier wall contact) and its shape at "
           "every frame is the solver state after one env step; the proximal shaft is the "
           "actuated base path (route centerline). Constant insertion command, no twist: the "
           "route is given and branch selection comes from the pre-shaped tip and contact, not "
           "a learned policy. Fluoroscopy: Lumen DRR of the contrast-filled lumen plus an "
           "analytic cylinder projection of the wire through the same C-arm, then Lumen's "
           "detector-realism model.",
           "commit": git_sha(), "lumen_tree_dirty": git_dirty(),
           "command": "python -m tools.xseries.clip_hero " + " ".join(sys.argv[1:] if argv is None else argv),
           "rollout_file": str(rpath),
           "scene": {"asset": "tools.xseries.anatomy.femoral_arch()", "target": a.target,
                     "env": "lumen.envs.tree_nav.TreeNavEnv", "max_insertion_mm_per_step": a.insertion,
                     "advance_speed_mm_s": a.insertion / meta["dt_s_per_step"], "seed": a.seed,
                     **meta},
           "result": {"steps": len(frames) - 1, "success": last.get("success"),
                      "safe_success": last.get("safe_success"),
                      "peak_wall_indentation_mm": max(pens) if pens else None,
                      "indentation_threshold_mm": meta.get("safety_max_pen_mm"),
                      "final_dist_to_target_mm": last.get("dist"), "sim_wall_s": meta.get("sim_wall_s")},
           "fluoro": cam.params, "versions": env_versions(), "outputs": {}}
    for k, d in dirs.items():
        mp4 = out / f"clip1-hero-{k}.mp4"
        rec["encode_cmd"] = encode(d, mp4)
        rec["outputs"][k] = {"file": mp4.name, "sha256": sha256(mp4), **probe(mp4)}
    write_manifest(out / "manifest.json", rec)
    print(rec["result"], rec["outputs"])


if __name__ == "__main__":
    main()
