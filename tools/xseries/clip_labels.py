"""Clip 2: synthetic fluoroscopy with its automatically generated CV labels.

Same episode as the hero clip. Left: the image. Right: the image with the labels Lumen
emits for free at every frame: vessel mask, device mask, device-tip keypoint.

    python -m tools.xseries.clip_labels
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.xseries.xs_common import (FluoroCam, arch_rollout, corner_tag, encode,
                                     env_versions, git_dirty, git_sha, gray_to_rgb, label,
                                     out_root, probe, sha256, write_manifest)

LZ = Image.Resampling.LANCZOS
VESSEL = np.array([64, 196, 210], float)
DEVICE = np.array([240, 84, 160], float)


def overlay(g, vmask, dmask):
    rgb = gray_to_rgb(g).astype(float)
    rgb[vmask] = 0.62 * rgb[vmask] + 0.38 * VESSEL
    rgb[dmask] = 0.15 * rgb[dmask] + 0.85 * DEVICE
    return rgb.clip(0, 255).astype(np.uint8)


def mask_rgb(m, color):
    out = np.zeros(m.shape + (3,), np.uint8)
    out[m] = color.astype(np.uint8)
    return out


def tip_marker(img: Image.Image, uv, nu, scale, text="tip"):
    d = ImageDraw.Draw(img)
    x, y = (uv[0] + 0.5) * scale, (nu - 1 - uv[1] + 0.5) * scale
    r = 16
    d.ellipse((x - r, y - r, x + r, y + r), outline=(250, 214, 80), width=3)
    d.line((x - r - 10, y, x - r + 4, y), fill=(250, 214, 80), width=3)
    d.line((x + r - 4, y, x + r + 10, y), fill=(250, 214, 80), width=3)
    from tools.xseries.xs_common import font
    d.text((x + r + 14, y - 14), text, font=font(22), fill=(250, 214, 80))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nu", type=int, default=720)
    ap.add_argument("--hold", type=int, default=50)
    ap.add_argument("--only", default="", help="comma frame indices: preview stills, no encode")
    a = ap.parse_args(argv)
    only = [int(x) for x in a.only.split(",") if x]

    out = Path(a.out) if a.out else out_root() / "clip2-autolabels"
    out.mkdir(parents=True, exist_ok=True)
    asset, target, frames, meta, rpath = arch_rollout("lcca_out", 0.45, seed=a.seed)
    cam = FluoroCam(asset, axis=(0.0, 1.0, 0.0), nu=a.nu, fov=1.0)
    vmask = cam.vessel_mask()          # anatomy and camera are static: project the vessel once
    dirs = {k: out / f"frames_{k}" for k in ("landscape", "square")}
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    seq = list(range(len(frames))) + [len(frames) - 1] * a.hold
    sample_labels = []
    for i, fi in enumerate(seq):
        fr = frames[fi]
        if only and i not in only:
            continue
        g = cam.image(fr.extra["wire"], seed=fi)
        m = {"device": cam.device_mask(fr.extra["wire"])}
        tip = cam.project(fr.nodes[-1])["uv"]
        gi, vm, dm = np.flipud(g), np.flipud(vmask), np.flipud(m["device"])
        raw = Image.fromarray(gray_to_rgb(gi)).resize((960, 960), LZ)
        lab = Image.fromarray(overlay(gi, vm, dm)).resize((960, 960), LZ)
        tip_marker(lab, tip, a.nu, 960 / a.nu)
        L = Image.new("RGB", (1920, 1080), (7, 9, 13))
        L.paste(raw, (0, 60)); L.paste(lab, (960, 60))
        label(L, "Synthetic fluoroscopy", (24, 18), 26)
        label(L, "Auto-labels:", (984, 18), 26)
        label(L, "vessel mask", (1150, 18), 26, tuple(int(c) for c in VESSEL))
        label(L, "device mask", (1318, 18), 26, tuple(int(c) for c in DEVICE))
        label(L, "tip keypoint", (1490, 18), 26, (250, 214, 80))
        corner_tag(L)
        L.save(dirs["landscape"] / f"{i:05d}.png")
        S = Image.new("RGB", (1080, 1080), (7, 9, 13))
        q = 540
        S.paste(raw.resize((q, q), LZ), (0, 0)); S.paste(lab.resize((q, q), LZ), (q, 0))
        S.paste(Image.fromarray(mask_rgb(vm, VESSEL)).resize((q, q), Image.Resampling.NEAREST), (0, q))
        S.paste(Image.fromarray(mask_rgb(dm, DEVICE)).resize((q, q), Image.Resampling.NEAREST), (q, q))
        for (x, y, t) in ((12, 10, "Image"), (q + 12, 10, "Image + labels"),
                          (12, q + 10, "Vessel mask"), (q + 12, q + 10, "Device mask")):
            label(S, t, (x, y), 22, bg=(10, 13, 18))
        corner_tag(S)
        S.save(dirs["square"] / f"{i:05d}.png")
        if fi in (0, len(frames) // 2, len(frames) - 1) and i < len(frames):
            sample_labels.append({"step": fr.info["step"], "tip_uv": tip,
                                  "device_mask_px": int(m["device"].sum()),
                                  "vessel_mask_px": int(vmask.sum())})
        if i % 50 == 0:
            print(f"frame {i}/{len(seq)}", flush=True)
    if only:
        print("preview written")
        return

    rec = {"clip": "2-autolabels",
           "description": "Lumen synthetic fluoroscopy and the labels its sensor emits with every "
           "frame: vessel mask (projected asset lumen), device mask (projected device polyline at "
           "its physical radius, floored at 1 px where that projects below one pixel), device-tip keypoint. Labels are exact by construction because "
           "they come from the same solver state as the image; no annotator or model produced them.",
           "commit": git_sha(), "lumen_tree_dirty": git_dirty(),
           "command": "python -m tools.xseries.clip_labels " + " ".join(sys.argv[1:] if argv is None else argv),
           "rollout_file": str(rpath), "label_samples": sample_labels,
           "scene": {"asset": "tools.xseries.anatomy.femoral_arch()", "target": "lcca_out",
                     "max_insertion_mm_per_step": 0.45, "seed": a.seed, **meta},
           "fluoro": cam.params, "device_mask": {"radius_mm": 0.2, "min_radius_px": 1.0,
                                                 "note": "projected radius floored at 1 px"},
           "versions": env_versions(), "outputs": {}}
    for k, d in dirs.items():
        mp4 = out / f"clip2-autolabels-{k}.mp4"
        rec["encode_cmd"] = encode(d, mp4)
        rec["outputs"][k] = {"file": mp4.name, "sha256": sha256(mp4), **probe(mp4)}
    write_manifest(out / "manifest.json", rec)
    print(rec["outputs"])


if __name__ == "__main__":
    main()
