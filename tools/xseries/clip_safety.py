"""Clip 3: wall-contact safety. Two runs reach the same target (brachiocephalic trunk)
in the same procedural arch. One pushes fast and indents the wall past the safety
threshold; the other advances slowly and stays under it. Both count as raw successes;
only one is a safe success. Frames are synchronized in simulated time.

    python -m tools.xseries.clip_safety
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.xseries.xs_common import (View3D, arch_rollout, corner_tag, encode, env_versions,
                                     font, git_dirty, git_sha, label, out_root, probe, sha256,
                                     write_manifest)

LZ = Image.Resampling.LANCZOS
DT = 0.02  # simulated seconds per env step (5e-3 s x 4 substeps); checked against the env


def run(ins, seed):
    def extra(env):
        return {"pen": np.asarray(env.sim.surface_penetration(), float)}
    return arch_rollout("lcca_out", ins, seed=seed, extra_fn=extra)


def chart(w, h, t_now, series, thr, tmax, ymax):
    img = Image.new("RGB", (w, h), (11, 14, 19))
    d = ImageDraw.Draw(img)
    L, R, T, B = 70, 20, 34, 40
    pw, ph = w - L - R, h - T - B
    d.rectangle((L, T, L + pw, T + ph), outline=(52, 62, 76))
    y_thr = T + ph * (1 - thr / ymax)
    d.line((L, y_thr, L + pw, y_thr), fill=(236, 185, 88), width=2)
    d.text((L + pw - 6, y_thr - 6), f"safety threshold {thr:.1f} mm (= contact barrier width)", font=font(18),
           fill=(236, 185, 88), anchor="rd")
    for yv in np.arange(0, ymax + 1e-9, 0.2):
        yy = T + ph * (1 - yv / ymax)
        d.text((L - 8, yy), f"{yv:.1f}", font=font(15), fill=(130, 140, 152), anchor="rm")
    d.text((8, T - 26), "Wall indentation, deepest device node (mm)", font=font(18), fill=(190, 198, 208))
    d.text((L, T + ph + 22), f"simulated time   t = {t_now:4.2f} s", font=font(16),
           fill=(130, 140, 152), anchor="lm")
    for vals, color in series:
        pts = [(L + pw * (k * DT) / tmax, T + ph * (1 - min(v, ymax) / ymax))
               for k, v in enumerate(vals) if k * DT <= t_now + 1e-9]
        if len(pts) > 1:
            d.line(pts, fill=color, width=3)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", type=float, default=0.9, help="mm per step, aggressive run")
    ap.add_argument("--slow", type=float, default=0.45, help="mm per step, careful run")
    ap.add_argument("--hold", type=int, default=50)
    ap.add_argument("--only", default="", help="comma frame indices: preview stills, no encode")
    a = ap.parse_args(argv)
    only = [int(x) for x in a.only.split(",") if x]

    out = Path(a.out) if a.out else out_root() / "clip3-wall-safety"
    out.mkdir(parents=True, exist_ok=True)
    asset, target, fast, fmeta, fpath = run(a.fast, a.seed)
    _, _, slow, smeta, spath = run(a.slow, a.seed)
    assert abs(fmeta["dt_s_per_step"] - DT) < 1e-12, fmeta
    thr = float(fmeta["safety_max_pen_mm"])

    def summary(fr, ins):
        pens = [f.info.get("max_pen", 0.0) for f in fr[1:]]
        loads = [f.info.get("wall_load_max", 0.0) for f in fr[1:]]
        hit = [k for k, f in enumerate(fr[1:], 1) if f.info.get("unsafe")]
        return {"insertion_mm_per_step": ins, "insertion_speed_mm_s": ins / DT,
                "steps": len(fr) - 1, "time_to_target_s": round((len(fr) - 1) * DT, 2),
                "success": fr[-1].info.get("success"),
                "safe_success": bool(fr[-1].info.get("success") and max(pens) <= thr),
                "peak_indentation_mm": round(max(pens), 4),
                "first_unsafe_step": hit[0] if hit else None,
                "peak_wall_load_sim_units": round(max(loads), 2)}

    sf, ss = summary(fast, a.fast), summary(slow, a.slow)
    print("fast", sf, "\nslow", ss, flush=True)
    if not (sf["success"] and not sf["safe_success"] and ss["success"] and ss["safe_success"]):
        raise SystemExit("pair does not show reach-but-unsafe vs safe; adjust --fast/--slow")

    view = View3D(asset, size_px=(960, 760), elev=10.0, azim0=-90.0, zoom=1.9)
    view.c = view.c + np.array([0.0, 0.0, 22.0])     # frame the arch and the branches
    n_frames = max(len(fast), len(slow)) + a.hold
    tmax = (max(len(fast), len(slow)) - 1) * DT
    ymax = max(0.8, 1.15 * max(sf["peak_indentation_mm"], ss["peak_indentation_mm"]))
    red, green = (232, 90, 92), (112, 211, 141)
    dirs = {k: out / f"frames_{k}" for k in ("landscape", "square")}
    for d in dirs.values():
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    def panel(fr, k, color):
        f = fr[min(k, len(fr) - 1)]
        pen = f.extra.get("pen")
        heat = (f.nodes, pen, thr) if pen is not None else None
        return Image.fromarray(view.render(f.extra["wire"], target=target, heat=heat,
                                           device_color="#5ce8ee")), f

    for i in range(n_frames):
        if only and i not in only:
            continue
        t_now = min(i, max(len(fast), len(slow)) - 1) * DT
        pf, ff = panel(fast, i, red)
        ps, fs = panel(slow, i, green)
        L = Image.new("RGB", (1920, 1080), (7, 9, 13))
        L.paste(pf, (0, 0)); L.paste(ps, (960, 0))
        for x0, fr_, s, col, name in ((0, ff, sf, red, "Fast push"), (960, fs, ss, green, "Slow advance")):
            reached = fr_.info.get("success")
            pk = max(f.info.get("max_pen", 0.0) for f in (fast if x0 == 0 else slow)[1:fr_.info["step"] + 1]) if fr_.info["step"] else 0.0
            label(L, f"{name}  {s['insertion_speed_mm_s']:.0f} mm/s", (x0 + 24, 18), 26, col)
            label(L, f"indentation now {fr_.info.get('max_pen', 0.0):.2f} mm   peak {pk:.2f} mm",
                  (x0 + 24, 56), 20, (190, 198, 208))
            if reached:
                verdict = "target reached, SAFE" if pk <= thr else "target reached, UNSAFE"
                label(L, verdict, (x0 + 24, 700), 28, green if pk <= thr else red, bg=(10, 13, 18))
        L.paste(chart(1920, 320, t_now, [([f.info.get("max_pen", 0.0) for f in fast], red),
                                         ([f.info.get("max_pen", 0.0) for f in slow], green)],
                      thr, tmax, ymax), (0, 760))
        corner_tag(L)
        L.save(dirs["landscape"] / f"{i:05d}.png")
        S = Image.new("RGB", (1080, 1080), (7, 9, 13))
        S.paste(pf.crop((150, 40, 810, 700)).resize((540, 540), LZ), (0, 40))
        S.paste(ps.crop((150, 40, 810, 700)).resize((540, 540), LZ), (540, 40))
        label(S, "Fast push", (16, 8), 24, red); label(S, "Slow advance", (556, 8), 24, green)
        for x0, fr_, src in ((0, ff, fast), (540, fs, slow)):
            if fr_.info.get("success"):
                pk = max(f.info.get("max_pen", 0.0) for f in src[1:])
                label(S, "reached, SAFE" if pk <= thr else "reached, UNSAFE", (x0 + 16, 540),
                      24, green if pk <= thr else red, bg=(10, 13, 18))
        S.paste(chart(1080, 440, t_now, [([f.info.get("max_pen", 0.0) for f in fast], red),
                                         ([f.info.get("max_pen", 0.0) for f in slow], green)],
                      thr, tmax, ymax), (0, 600))
        corner_tag(S)
        S.save(dirs["square"] / f"{i:05d}.png")
        if i % 40 == 0:
            print(f"frame {i}/{n_frames}", flush=True)
    if only:
        print("preview written")
        return

    rec = {"clip": "3-wall-safety",
           "description": "Two guidewire runs from the descending aorta to the left common "
           "carotid in the same procedural aortic arch, differing only in insertion speed. Lumen's safety endpoint is geometric "
           "wall indentation: how far the device surface overlaps the vessel wall surface, in mm "
           "(TreeNavEnv.safety_max_pen). A run is a safe success only if it reaches the target "
           "without ever exceeding the threshold. Device nodes are colored by their current "
           "indentation. Wall load is reported in simulator units only; Lumen does not claim "
           "Newtons (no SI force calibration exists in the open core).",
           "threshold": {"endpoint": "peak device-surface/wall overlap over the episode",
                         "value_mm": thr, "units": "mm (geometric)",
                         "source": "lumen.envs.tree_nav.TreeNavEnv default safety_max_pen",
                         "rationale": "Equals the solver's contact barrier width (d_hat = 0.3 mm). "
                         "Overlap beyond the barrier band means the wire is being driven into the "
                         "wall instead of sliding along it. Geometric proxy, not a force: Lumen's "
                         "open core reports wall load in uncalibrated simulator units and ships no "
                         "Newton conversion (lumen/safety.py)."},
           "commit": git_sha(), "lumen_tree_dirty": git_dirty(),
           "command": "python -m tools.xseries.clip_safety " + " ".join(sys.argv[1:] if argv is None else argv),
           "rollout_files": [str(fpath), str(spath)], "fast": sf, "slow": ss, "sim": fmeta,
           "versions": env_versions(), "outputs": {}}
    for k, d in dirs.items():
        mp4 = out / f"clip3-wall-safety-{k}.mp4"
        rec["encode_cmd"] = encode(d, mp4)
        rec["outputs"][k] = {"file": mp4.name, "sha256": sha256(mp4), **probe(mp4)}
    write_manifest(out / "manifest.json", rec)
    print(rec["outputs"])


if __name__ == "__main__":
    main()
