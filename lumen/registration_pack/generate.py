"""Procedural generation of the ``full-tree-v1`` registration case (layer 1).

Renders a synthetic asymmetric multilevel nonplanar curved branching vessel tree
through a true :class:`lumen.sensors.carm.CArm` (divergent-beam source at the
origin, detector axes +x right / +y down / +z forward, screen y down after
flipping CArm's native projected row as ``nv-1-v``) and emits::

    <case_dir>/problem/{image.png, geometry.json, problem.json}
    <case_dir>/answer/{pose.json, overlay.png, territories.json, mask_*.png}

Everything is millimetres. Geometry points are stored in *model* space (pre
pose); the ground-truth rigid pose lives only under ``answer/``. Generation is
deterministic for the versioned geometry recipe and the packaged answer pose.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

from lumen.sensors.carm import CArm

CASE_ID = "full-tree-v1"
PROVENANCE = "procedural"

# Camera / detector (mm, pixels)
WIDTH_MM = HEIGHT_MM = 180.0
NU = NV = 512
SDD_MM = 600.0
OBJECT_DEPTH_MM = 300.0  # where the pose places the tree root
MU_PER_MM = 0.14  # linear attenuation used for the DRR
SS = 2  # supersampling factor for anti-aliasing

TERRITORY_RGB = {
    "inlet": (106, 70, 132),
    "west": (195, 69, 43),
    "apical": (28, 140, 109),
    "east": (46, 103, 173),
}
TERRITORY_NAMES = tuple(TERRITORY_RGB)


def _json(path, obj):
    path.write_bytes((json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n").encode())


def edge_overlay(image, mask):
    """Paint the four-connected inner boundary on the observed gray image."""
    p = np.pad(mask, 1)
    edge = mask & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    overlay = np.repeat(image[:, :, None], 3, axis=2)
    overlay[edge] = (0, 120, 190)
    return overlay


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, arr: np.ndarray) -> None:
    """Deterministic 8-bit PNG (gray for 2-D, RGB for HxWx3); zlib level 9."""
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        color, channels = 0, 1
        raw = arr.astype(np.uint8, copy=False).tobytes()
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        color, channels = 2, 3
        raw = arr.astype(np.uint8, copy=False).tobytes()
    else:
        raise ValueError(f"unsupported PNG array shape {arr.shape}")
    stride = w * channels
    scan = b"".join(b"\x00" + raw[y * stride : (y + 1) * stride] for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0)
    idat = zlib.compress(scan, 9)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def read_png(path: Path) -> np.ndarray:
    """Read back a PNG written by :func:`write_png` (gray -> HxW, RGB -> HxWx3)."""
    blob = path.read_bytes()
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, meta = 8, b"", None
    while pos < len(blob):
        (length,) = struct.unpack(">I", blob[pos : pos + 4])
        tag = blob[pos + 4 : pos + 8]
        data = blob[pos + 8 : pos + 8 + length]
        if tag == b"IHDR":
            meta = struct.unpack(">IIBBBBB", data)
        elif tag == b"IDAT":
            idat += data
        pos += 12 + length
    w, h, depth, color, _, _, _ = meta
    assert depth == 8
    channels = 1 if color == 0 else 3
    raw = zlib.decompress(idat)
    stride = w * channels
    out = np.empty((h, stride), np.uint8)
    for y in range(h):
        out[y] = np.frombuffer(raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)], np.uint8)
    arr = out.reshape(h, w, channels)
    return arr[:, :, 0] if channels == 1 else arr


# ----------------------------------------------------------------- tree -----

# Each procedural cubic starts at its parent's tip; controls and tips are model mm.
# Unequal lengths, depth excursions and daughter counts break mirror symmetry.
CURVES = [
    (None, "inlet", 2.2, (2, 29, 2), (-5, 16, -2), (-2, 10, 0)),
    (0, "west", 1.55, (-5, 2, -3), (-16, 9, -7), (-19, -2, -8)),
    (1, "west", 1.05, (-23, -7, -8), (-18, -14, -6), (-22, -19, -5)),
    (2, "west", 0.65, (-24, -23, -4), (-30, -23, -1), (-31, -30, 1)),
    (2, "west", 0.8, (-20, -24, -6), (-24, -29, -10), (-19, -33, -9)),
    (1, "west", 0.85, (-23, -4, -8), (-30, 2, -10), (-35, -7, -14)),
    (0, "apical", 1.35, (1, 5, 5), (-7, -1, 10), (-4, -10, 14)),
    (6, "apical", 0.9, (-2, -18, 19), (-8, -19, 18), (-6, -29, 21)),
    (7, "apical", 0.55, (-8, -33, 22), (-10, -34, 23), (-9, -37, 24)),
    (7, "apical", 0.6, (-5, -34, 19), (-2, -31, 20), (0, -36, 18)),
    (6, "apical", 0.75, (1, -14, 12), (1, -17, 14), (5, -23, 9)),
    (0, "east", 1.65, (6, 3, 0), (12, 10, -4), (16, 0, -6)),
    (11, "east", 1.0, (20, -7, -8), (14, -13, -11), (20, -20, -15)),
    (12, "east", 0.68, (24, -24, -15), (21, -27, -13), (25, -30, -14)),
    (12, "east", 0.62, (18, -24, -18), (14, -22, -16), (13, -28, -17)),
    (11, "east", 0.86, (22, -4, -4), (28, 4, 2), (33, -5, 4)),
]


def build_tree():
    t = np.linspace(0, 1, 25)[:, None]
    branches = []
    for i, (parent, territory, radius, a, b, tip) in enumerate(CURVES):
        start = np.array([1, 38, 0] if parent is None else CURVES[parent][-1])
        points = (
            (1 - t) ** 3 * start + 3 * (1 - t) ** 2 * t * a + 3 * (1 - t) * t**2 * b + t**3 * tip
        )
        branches.append(
            dict(
                id=f"b{i:02}",
                parent=None if parent is None else f"b{parent:02}",
                territory=territory,
                radius_mm=radius,
                points_mm=points.round(4).tolist(),
            )
        )
    return branches


def make_carm() -> CArm:
    """True C-arm: source at the origin looking down +z, up=(0,-1,0)."""
    return CArm(
        source=np.zeros(3),
        detector_center=np.array([0.0, 0.0, SDD_MM]),
        up=np.array([0.0, -1.0, 0.0]),
        width=WIDTH_MM,
        height=HEIGHT_MM,
        nu=NU * SS,
        nv=NV * SS,
    )


def _camera_params():
    return dict(
        fx_px=NU * SDD_MM / WIDTH_MM,
        fy_px=NV * SDD_MM / HEIGHT_MM,
        cx_px=NU / 2.0 - 0.5,
        cy_px=NV / 2.0 - 0.5,
        width_px=NU,
        height_px=NV,
        axes="+x right, +y down, +z forward (right-handed, source at origin, screen y down)",
        source_to_detector_mm=SDD_MM,
    )


def render(branches: list[dict], pose: np.ndarray):
    """Divergent-beam DRR via CArm.project + the ``nv-1-v`` screen flip.

    Returns (image HxW uint8, overlay HxWx3 uint8, masks {territory: HxW bool}).
    """
    carm = make_carm()
    big = NU * SS
    attn = np.zeros((NV * SS, big), float)
    terr_attn = np.zeros((len(TERRITORY_NAMES), NV * SS, big), float)
    terr_idx = {t: i for i, t in enumerate(TERRITORY_NAMES)}
    mm_per_px = WIDTH_MM * OBJECT_DEPTH_MM / SDD_MM / big  # at object depth

    for b in branches:
        pts = np.asarray(b["points_mm"], float)
        cam = (pose[:3, :3] @ pts.T).T + pose[:3, 3]
        uv = np.array([carm.project(p) for p in cam])  # (u, v) native, v up
        cols, rows = uv[:, 0], (big - 1) - uv[:, 1]  # flip: screen y down
        r0 = b["radius_mm"]
        for i in range(len(pts) - 1):
            x0, y0, x1, y1 = cols[i], rows[i], cols[i + 1], rows[i + 1]
            if not all(np.isfinite([x0, y0, x1, y1])):
                continue
            depth = 0.5 * (cam[i, 2] + cam[i + 1, 2])
            mag = SDD_MM / max(depth, 1e-6)
            r_mm = r0 * (1.0 - 0.4 * i / max(1, len(pts) - 2))  # gentle taper
            r_px = max(1.0, r_mm * mag * big / WIDTH_MM)

            ti = terr_idx[b["territory"]]
            _paint(attn, terr_attn[ti], x0, y0, x1, y1, r_px, mm_per_px)

    # downsample (box) and convert attenuation to grayscale
    attn = attn.reshape(NV, SS, NU, SS).mean(axis=(1, 3))
    img = np.clip(255.0 * np.exp(-MU_PER_MM * attn), 0, 255).astype(np.uint8)

    terr_attn = terr_attn.reshape(len(TERRITORY_NAMES), NV, SS, NU, SS).mean(axis=(2, 4))
    claimed = terr_attn.sum(axis=0) > 1e-9
    terr_map = terr_attn.argmax(axis=0)

    masks = {}
    for t, i in terr_idx.items():
        masks[t] = claimed & (terr_map == i)

    # overlay: grayscale base with territory-tinted vessels
    base = np.stack([img, img, img], axis=2)
    overlay = base.copy()
    for t, i in terr_idx.items():
        sel = masks[t] & (attn > 0.05)
        color = np.asarray(TERRITORY_RGB[t], float)
        overlay[sel] = (0.35 * overlay[sel] + 0.65 * color).astype(np.uint8)
    return img, overlay, masks


def _paint(attn: np.ndarray, terr: np.ndarray, x0, y0, x1, y1, r_px, mm_per_px: float) -> None:
    """Accumulate ray path length through one cylinder segment (screen px)."""
    h, w = attn.shape
    xa, xb = int(max(0, min(x0, x1) - r_px - 1)), int(min(w, max(x0, x1) + r_px + 2))
    ya, yb = int(max(0, min(y0, y1) - r_px - 1)), int(min(h, max(y0, y1) + r_px + 2))
    if xa >= xb or ya >= yb:
        return
    ys, xs = np.mgrid[ya:yb, xa:xb]
    dx, dy = x1 - x0, y1 - y0
    ll = dx * dx + dy * dy
    t = 0.0 if ll < 1e-12 else np.clip(((xs - x0) * dx + (ys - y0) * dy) / ll, 0, 1)
    d2 = (xs - x0 - t * dx) ** 2 + (ys - y0 - t * dy) ** 2
    r2 = r_px * r_px
    inside = d2 < r2
    if not inside.any():
        return
    # chord length of the ray through the cylinder, per pixel footprint
    chord = 2.0 * np.sqrt(np.clip(r2 - d2, 0, None)) * mm_per_px
    add = np.where(inside, chord, 0.0)
    attn[ya:yb, xa:xb] += add
    terr[ya:yb, xa:xb] += add


# ------------------------------------------------------------ pack output ---


def generate_case(case_dir: Path | str) -> Path:
    """Rebuild the one frozen case; never overwrite an existing destination."""
    case_dir = Path(case_dir)
    answer_pose = (Path(__file__).parent / CASE_ID / "answer" / "pose.json").read_bytes()
    T = np.array(json.loads(answer_pose)["model_to_camera_mm"])
    branches = build_tree()
    img, overlay, masks = render(branches, T)
    case_dir.mkdir(parents=True, exist_ok=False)
    problem, answer = case_dir / "problem", case_dir / "answer"
    problem.mkdir()
    answer.mkdir()
    write_png(problem / "image.png", img)
    (problem / "geometry.json").write_text(
        json.dumps({"branches": branches}, separators=(",", ":")) + "\n"
    )
    cam = _camera_params()
    (problem / "problem.json").write_text(
        json.dumps(
            {
                "schema": "lumen-registration/1",
                "case_id": CASE_ID,
                "provenance": PROVENANCE,
                "geometry": "geometry.json",
                "image": "image.png",
                "camera": cam,
            },
            indent=1,
        )
        + "\n"
    )

    (answer / "pose.json").write_bytes(answer_pose)
    terr_map = {t: f"mask_{t}.png" for t, m in masks.items() if m.any()}
    for t, fname in terr_map.items():
        write_png(answer / fname, (masks[t] * 255).astype(np.uint8))
    (answer / "territories.json").write_text(json.dumps(terr_map, indent=1) + "\n")
    write_png(answer / "overlay.png", overlay)
    return case_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lumen.registration_pack.generate", description=__doc__)
    ap.add_argument("case_dir")
    args = ap.parse_args(argv)
    out = generate_case(args.case_dir)
    print(f"wrote {out}/problem and {out}/answer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
