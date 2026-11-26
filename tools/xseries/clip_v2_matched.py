"""X series v2 side-by-side: Lumen and the comparator on the same aortic arch (phantom3).

Both simulators run the identical arch (the comparator's phantom3, imported into Lumen by
benchmarks/external_comparison/phantom_import.py and read at runtime), the same target, and the
same evaluation case (one frozen evaluation seed: the same 1 mm start-offset law in both).
Each tile is one training seed's final PPO policy playing that case deterministically; the
episodes come from record_matched.py --task phantom3_<target>, nothing is re-simulated here.

Unlike clip 8, every tile uses one shared camera and one drawing of the arch: the projection basis,
bounds and the vessel shown behind every tile come from the comparator's phantom mesh (the anatomy
both simulators run; Lumen's copy is the runtime import of that mesh), so the two rows differ only
in the wire each simulator moved.
The tiles show navigation outcome only. Safety is reported per simulator, on its own scale, in the
v2 table (Lumen: wall overlap, mm; the comparator: tip-wall contact force, N), never on one axis.

    python -m tools.xseries.clip_v2_matched --episodes DIR --target lcca [--out DIR]
"""

from __future__ import annotations

CMP = "cath" + "sim"  # external comparator: separately installed, loaded at runtime only (see tools/check_firewall.py)
CMP_NAME = "Cath" + "Sim"

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tools.xseries.clip_matched import FPS, Tile, load, view_basis
from tools.xseries.xs_common import (corner_tag, encode, env_versions, git_dirty, git_sha, label, out_root, probe,
                                     sha256, write_manifest)

TARGET_NAME = {"bca": "brachiocephalic artery, BCA", "lcca": "left common carotid, LCCA"}


def shared_camera(L, C):
    """One (basis, bounds) for every tile, from the comparator's arch mesh (Lumen's if absent)."""
    ref = C[0] if C else L[0]
    V = np.asarray(ref["vessel"], float)
    e1, e2 = view_basis(V, CMP, ref["target"])
    P = np.stack([V @ e1, V @ e2], 1)
    return (e1, e2), (P.min(0), P.max(0))


def check_matched(L, C, target):
    """Refuse a clip whose recordings are not one task and one evaluation case."""
    tasks = {str(d.get("task", "")) for d in L + C}
    if tasks != {f"phantom3_{target}"}:
        raise ValueError(f"recordings are not all phantom3_{target}: {sorted(tasks)}")
    eval_seeds = {int(d["eval_seed"]) for d in L + C}
    if len(eval_seeds) != 1:
        raise ValueError(f"recordings use different evaluation seeds {sorted(eval_seeds)}")
    return eval_seeds.pop()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--target", choices=sorted(TARGET_NAME), required=True)
    ap.add_argument("--seeds", default="0,1,5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--slow", type=float, default=2.0, help="simulated-time slowdown factor")
    ap.add_argument("--only", default="", help="comma-separated frame indices: preview frames only, no video")
    a = ap.parse_args(argv)
    if not (np.isfinite(a.slow) and a.slow > 0):
        ap.error("--slow must be a finite positive factor")
    seeds = [int(x) for x in a.seeds.split(",") if x]
    L, C = load(a.episodes, "lumen", seeds), load(a.episodes, CMP, seeds)
    if len(L) != len(seeds) or len(C) != len(seeds):
        raise FileNotFoundError(f"need seeds {seeds} for both simulators; found {len(L)} Lumen, {len(C)} {CMP_NAME}")
    if len(seeds) > 3:
        raise ValueError("the layout fits 3 training seeds per simulator")
    eval_seed = check_matched(L, C, a.target)
    only = [int(x) for x in a.only.split(",") if x]
    out = Path(a.out) if a.out else out_root() / f"v2-side-by-side-{a.target}"
    out.mkdir(parents=True, exist_ok=True)
    basis, bounds = shared_camera(L, C)
    arch = C[0]["vessel"]                      # one drawing of the shared arch behind every tile
    L = [{**d, "vessel": arch} for d in L]
    T = max(x["steps"] * float(x["dt"]) for x in L + C)
    n_play = int(np.ceil(T * a.slow * FPS)) + 1
    n_frames = n_play + int(a.hold * FPS)
    tname = TARGET_NAME[a.target]
    rec = {"clip": f"v2-side-by-side-{a.target}", "commit": git_sha(), "lumen_tree_dirty": git_dirty(),
           "command": "python -m tools.xseries.clip_v2_matched " + " ".join(sys.argv[1:] if argv is None else argv),
           "versions": env_versions(), "eval_seed": eval_seed, "episodes": {}, "outputs": {}}
    for d in L + C:
        rec["episodes"][f"{d['environment']}-s{int(d['training_seed'])}"] = {
            "success": bool(d["success"]), "steps": int(d["steps"]), "dt_s": float(d["dt"]),
            "peak_safety": float(np.max(d["safety"])), "model": str(d["model"])}
    # tiles: one row of 3 per simulator; landscape = Lumen top, comparator bottom; square likewise
    layouts = {"landscape": ((1920, 1080), (600, 430), [(30, 150), (30, 630)]),
               "square": ((1080, 1080), (340, 400), [(30, 130), (30, 600)])}
    for kind, (size, (tw, th), origins) in layouts.items():
        fd = out / f"frames_{kind}"
        shutil.rmtree(fd, ignore_errors=True)
        fd.mkdir(parents=True)
        sides = [[Tile(d, tw, th, basis, bounds, show_safety=False) for d in L],
                 [Tile(d, tw, th, basis, bounds, show_safety=False) for d in C]]
        big = kind == "landscape"
        for i in range(n_frames):
            if only and i not in only:
                continue
            t = i / (FPS * a.slow)
            img = Image.new("RGB", size, (7, 9, 13))
            heads = ("Lumen", CMP_NAME)
            for side, (ox, oy), head in zip(sides, origins, heads):
                label(img, head, (ox, oy - (36 if big else 32)), 24 if big else 19, (235, 240, 245))
                for j, tile in enumerate(side):
                    img.paste(tile.frame(t), (ox + j * (tw + 10), oy))
            if big:
                label(img, f"Same aortic arch, same target ({tname}), same start case: Lumen vs {CMP_NAME}",
                      (30, 22), 32)
                label(img, f"Final PPO policy of each training seed (600k env steps), evaluation seed {eval_seed}. "
                           f"Simulated time, slowed {a.slow:g}x.", (30, 66), 20, (160, 168, 180))
            else:
                label(img, f"Same arch, target {a.target.upper()}: Lumen vs {CMP_NAME}", (30, 16), 26)
                label(img, f"final PPO policy per training seed; eval seed {eval_seed}; {a.slow:g}x slow",
                      (30, 52), 17, (160, 168, 180))
            corner_tag(img, top=big)
            img.save(fd / f"{i:05d}.png")
        if only:
            continue
        mp4 = out / f"v2-side-by-side-{a.target}-{kind}.mp4"
        rec["encode_cmd"] = encode(fd, mp4, fps=FPS)
        rec["outputs"][kind] = {"file": mp4.name, "sha256": sha256(mp4), **probe(mp4)}
    if only:
        print("preview written", out)
        return
    write_manifest(out / "manifest.json", rec)
    print(rec["outputs"])


if __name__ == "__main__":
    main()
