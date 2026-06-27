"""Tiny preview exporters for observation arrays.

Pure stdlib on purpose: examples/tests can emit inspectable PNGs and an AVI without
bringing PIL/imageio into the core package.
"""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np


def _u8(arr):
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.asarray(a, float)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = 255.0 * (a - lo) / (hi - lo + 1e-12)
    return np.clip(a, 0, 255).astype(np.uint8)


def write_png(path, arr) -> None:
    """Write a grayscale or RGB PNG."""
    a = np.ascontiguousarray(_u8(arr))
    if a.ndim == 2:
        h, w = a.shape
        color = 0
        row = lambda r: a[r].tobytes()
    elif a.ndim == 3 and a.shape[2] == 3:
        h, w = a.shape[:2]
        color = 2
        row = lambda r: a[r].tobytes()
    else:
        raise ValueError("PNG preview expects (H,W) grayscale or (H,W,3) RGB")

    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)

    raw = b"".join(b"\x00" + row(r) for r in range(h))
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9))
                + chunk(b"IEND", b""))


def _rgb24(frame):
    a = _u8(frame)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("AVI preview expects grayscale or RGB frames")
    # DIB rows are BGR and bottom-up, padded to 4-byte boundaries.
    bgr = a[:, :, ::-1][::-1]
    pad = (-bgr.shape[1] * 3) % 4
    rows = []
    for r in bgr.reshape(a.shape[0], -1):
        rows.append(r.tobytes() + b"\x00" * pad)
    return b"".join(rows)


def _chunk(tag, data):
    return tag + struct.pack("<I", len(data)) + data + (b"\x00" if len(data) % 2 else b"")


def _list(tag, data):
    return b"LIST" + struct.pack("<I", len(data) + 4) + tag + data


def write_avi(path, frames, fps: int = 10) -> None:
    """Write an uncompressed RGB AVI preview.

    This is intentionally minimal but standard enough for common players and CI
    validation. For large corpora, use imageio/ffmpeg externally.
    """
    frames = [np.asarray(f) for f in frames]
    if not frames:
        raise ValueError("AVI preview needs at least one frame")
    first = _u8(frames[0])
    h, w = first.shape[:2]
    if any(_u8(f).shape[:2] != (h, w) for f in frames[1:]):
        raise ValueError("all AVI frames must have the same shape")
    rgb = [_rgb24(f) for f in frames]
    frame_size = len(rgb[0])
    fps = int(fps)
    if fps <= 0:
        raise ValueError("fps must be positive")

    usec_per_frame = int(round(1_000_000 / fps))
    movi = b"".join(_chunk(b"00db", f) for f in rgb)
    movi_list = _list(b"movi", movi)

    avih = struct.pack("<IIIIIIIIII4I",
                       usec_per_frame, frame_size * fps, 0, 0x10, len(rgb), 0, 1,
                       frame_size, w, h, 0, 0, 0, 0)
    strh = struct.pack("<4s4s14I",
                       b"vids", b"DIB ", 0, 0, 0, 0, 1, fps, 0, len(rgb),
                       frame_size, 0xffffffff, 0, 0, 0, w | (h << 16))
    strf = struct.pack("<IIIHHIIIIII",
                       40, w, h, 1, 24, 0, frame_size, 0, 0, 0, 0)
    hdrl = _list(b"hdrl", _chunk(b"avih", avih)
                 + _list(b"strl", _chunk(b"strh", strh) + _chunk(b"strf", strf)))
    body = hdrl + movi_list
    riff = b"RIFF" + struct.pack("<I", len(body) + 4) + b"AVI " + body
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(riff)
