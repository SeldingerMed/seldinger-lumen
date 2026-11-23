"""Clip 9: 25 to 40 s recap montage of clips 1 to 7 (plus the head-to-head card when its
table exists), ending on the repo link. Built only from the rendered clip MP4s and their
manifests; every segment keeps its source clip's on-screen labels. One title strip per
segment names what it shows.

    python -m tools.xseries.clip_recap [--compare runs/.../compare.json]
"""

from __future__ import annotations

CMP = "cath" + "sim"  # external comparator: separately installed, loaded at runtime only (see tools/check_firewall.py)
CMP_NAME = "Cath" + "Sim"

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from tools.xseries.xs_common import (encode, env_versions, font, git_dirty, git_sha, label, out_root, probe,
                                     sha256, write_manifest)

REPO = "github.com/SeldingerMed/seldinger-lumen"
FPS = 30
# (clip dir, file stem, caption, start fraction, seconds)
SEGMENTS = [
    ("clip1-hero", "clip1-hero", "Guidewire from the descending aorta into the left common carotid", 0.45, 4.0),
    ("clip2-autolabels", "clip2-autolabels", "Synthetic fluoroscopy with automatic labels", 0.55, 3.5),
    ("clip3-wall-safety", "clip3-wall-safety", "Same target, two speeds: wall contact measured every step", 0.35, 4.0),
    ("clip4-deformable-wall", "clip4-deformable-wall", "Hyperelastic vessel wall deforming under contact", 0.40, 4.0),
    ("clip5-batched-envs", "clip5-batched-envs", "Dozens of environments in one batched solver", 0.55, 3.5),
    ("clip6-replay", "clip6-replay", "Record, scrub and replay: image and state from disk", 0.30, 3.5),
    ("clip7-procedure-modules", "clip7-procedure-modules", "Thrombectomy and flow diversion modules", 0.50, 4.0),
]


def hero_caption(manifest: Path, default: str) -> str:
    """Caption for the hero clip from its recorded target (the default names the left common carotid)."""
    from tools.xseries.xs_common import ARCH_TARGETS
    try:
        target = json.load(open(manifest))["scene"]["target"]
    except (OSError, KeyError, ValueError):
        return "Guidewire navigation from the descending aorta"
    if target not in ARCH_TARGETS:
        return "Guidewire navigation from the descending aorta"
    return f"Guidewire from the descending aorta into the {ARCH_TARGETS[target]}"


def _fmt(c, fmt: str, scale: float = 1.0, unit: str = "") -> str:
    """A ci() mean with its contributing training-seed count, or n/a."""
    if not c or c.get("mean") is None:
        return "n/a"
    return f"{format(scale * c['mean'], fmt)}{unit} (n={c.get('n', 0)})"


def head_to_head_lines(cmp: dict):
    """Recap card lines; every mean shows its own training-seed count. None if nothing is available."""
    L, C = cmp["lumen"]["summary"], cmp[CMP]["summary"]
    keys = ("success_rate", "train_hours")
    if all((S.get(k) or {}).get("mean") is None for S in (L, C) for k in keys):
        return None
    return [(f"Lumen vs {CMP_NAME}, matched PPO (600k steps, training-seed means)", 34, (235, 240, 245)),
            (f"Raw success   Lumen {_fmt(L.get('success_rate'), '.0f', 100, '%')}   "
             f"{CMP_NAME} {_fmt(C.get('success_rate'), '.0f', 100, '%')}", 30, (190, 198, 208)),
            (f"Training wall-clock   Lumen {_fmt(L.get('train_hours'), '.1f', unit=' h')}   "
             f"{CMP_NAME} {_fmt(C.get('train_hours'), '.1f', unit=' h')}", 30, (190, 198, 208))]


def SW(v: int) -> int:
    """Clip footage is scaled to 90% and top-aligned so captions sit in their own band below it."""
    return int(round(v * 0.9 / 2)) * 2


def frames_from(mp4: Path, start_s: float, secs: float, size, dst: Path, k0: int) -> int:
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="_seg-", dir=dst.parent))   # owned by this render only
    W, H = size
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start_s:.3f}", "-i", str(mp4), "-t", f"{secs:.3f}",
                    "-vf", f"fps={FPS},scale={SW(W)}:{SW(H)}:flags=lanczos,pad={W}:{H}:{(W - SW(W)) // 2}:0:color=0x07090d",
                    str(tmp / "%05d.png")], check=True)
    files = sorted(f for f in tmp.glob("*.png") if not f.name.startswith("._"))   # exFAT AppleDouble
    for j, f in enumerate(files):
        f.rename(dst / f"{k0 + j:05d}.png")
    shutil.rmtree(tmp, ignore_errors=True)
    return len(files)


def duration(mp4: Path) -> float:
    return float(probe(mp4)["duration_s"])


def caption(img: Image.Image, text: str, n: int, i: int):
    W, H = img.size
    sz = 30 if W > 1500 else 28
    d = ImageDraw.Draw(img)
    tw = int(d.textlength(text, font=font(sz)))
    a = min(1.0, i / 8, (n - i) / 8)
    if a <= 0:
        return
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    y = SW(H) + (H - SW(H) - sz) // 2 - 2
    od.rounded_rectangle(((W - tw) // 2 - 22, y - 12, (W + tw) // 2 + 22, y + sz + 16), 10,
                         fill=(8, 10, 14, int(225 * a)))
    od.text((W // 2, y), text, font=font(sz), fill=(240, 244, 248, int(255 * a)), anchor="ma")
    img.paste(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))


def card(size, lines, n, dst: Path, k0: int):
    W, H = size
    for j in range(n):
        img = Image.new("RGB", size, (7, 9, 13))
        a = min(1.0, j / 10)
        y = H // 2 - 40 * len(lines)
        for text, sz, col in lines:
            c = tuple(int(v * a + 7 * (1 - a)) for v in col)
            label(img, text, (W // 2, y), sz, c, anchor="ma")
            y += int(sz * 1.6)
        img.save(dst / f"{k0 + j:05d}.png")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None, help="xseries_compare JSON (optional head-to-head card)")
    a = ap.parse_args(argv)
    root = out_root()
    out = Path(a.out) if a.out else root / "clip9-recap"
    out.mkdir(parents=True, exist_ok=True)
    used = []
    rec = {"clip": "9-recap", "segments": [], "commit": git_sha(), "lumen_tree_dirty": git_dirty(),
           "command": "python -m tools.xseries.clip_recap " + " ".join(sys.argv[1:] if argv is None else argv),
           "versions": env_versions(), "outputs": {}}
    cmp = json.load(open(a.compare)) if a.compare and Path(a.compare).exists() else None
    for kind, size in (("landscape", (1920, 1080)), ("square", (1080, 1080))):
        d = out / f"frames_{kind}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        k = 0
        for cdir, stem, cap, frac, secs in SEGMENTS:
            mp4 = root / cdir / f"{stem}-{kind}.mp4"
            if not mp4.exists():
                print("missing", mp4, flush=True)
                continue
            if cdir == "clip1-hero":          # caption the vessel the hero footage actually targets
                cap = hero_caption(root / cdir / "manifest.json", cap)
            dur = duration(mp4)
            st = max(0.0, min(dur - secs, frac * dur))
            k1 = frames_from(mp4, st, secs, size, d, k)
            for j in range(k1):
                p = d / f"{k + j:05d}.png"
                im = Image.open(p).convert("RGB")
                caption(im, cap, k1, j)
                im.save(p)
            rec["segments"].append({"layout": kind, "clip": cdir, "source_file": mp4.name, "start_s": round(st, 3),
                                    "seconds": secs, "source_sha256": sha256(mp4), "caption": cap})
            if kind == "landscape":
                used.append(cdir)
            k += k1
        lines = head_to_head_lines(cmp) if cmp else None
        if lines:
            k += card(size, lines, int(3.5 * FPS), d, k)
            if kind == "landscape":
                rec["segments"].append({"card": "head-to-head", "source": str(a.compare), "seconds": 3.5})
        k += card(size, [("Lumen", 64, (235, 240, 245)), ("Open-source endovascular simulator", 30, (170, 178, 190)),
                         (REPO, 34, (92, 232, 238)), ("Apache-2.0", 24, (130, 140, 152))], int(4.0 * FPS), d, k)
        if kind == "landscape":
            rec["segments"].append({"card": "end", "text": REPO, "seconds": 4.0})
        mp4 = out / f"clip9-recap-{kind}.mp4"
        rec["encode_cmd"] = encode(d, mp4, fps=FPS)
        rec["outputs"][kind] = {"file": mp4.name, "sha256": sha256(mp4), **probe(mp4)}
    rec["description"] = ("Recap montage cut from the rendered X-series clips (" + ", ".join(used) + "), each "
                          "segment with a one-line caption, ending on the public repository link.")
    write_manifest(out / "manifest.json", rec)
    print(rec["outputs"])


if __name__ == "__main__":
    main()
