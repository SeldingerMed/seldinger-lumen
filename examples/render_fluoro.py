"""Render a guidewire to synthetic fluoroscopy (Layer 1 L1.0) and save a PNG.

    python examples/render_fluoro.py [out.png]

Pure numpy + stdlib PNG writer (no matplotlib/PIL). Uses a synthetic curved wire; to
render a live device, pass NewtonGuidewireSim.body_positions() as `nodes`.
"""

from __future__ import annotations

import struct
import sys
import zlib

import numpy as np

from lumen.sensors import FluoroSensor


def write_png(path, gray_u8):
    h, w = gray_u8.shape
    raw = b"".join(b"\x00" + gray_u8[r].tobytes() for r in range(h))
    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "fluoro.png"
    a = np.linspace(0, np.pi / 2, 40)
    wire = np.stack([30 * np.sin(a) + 3 * np.sin(6 * a), 2 * np.cos(3 * a),
                     30 * (1 - np.cos(a))], axis=1)
    # render the DRR line integral A (device = high attenuation); min-max is the
    # standard DRR display (A has no fixed range) and shows the device BRIGHT. For the
    # clinical Beer-Lambert look (dark device on a bright field) use beer_lambert=True
    # and scale I*255 directly (I is already in [0,1]).
    vessel = np.stack([30 * np.sin(a), np.zeros_like(a),
                       30 * (1 - np.cos(a))], axis=1)
    A, _ = FluoroSensor(mu_device=1.2, res=96, n_samples=260).render(
        wire, radius=0.6, contrast_nodes=vessel, contrast_radius=2.0, mu_contrast=0.16)
    u8 = (255 * (A - A.min()) / (float(A.max() - A.min()) + 1e-9)).astype(np.uint8)
    write_png(out, np.ascontiguousarray(np.flipud(u8)))
    print(f"wrote {out}  ({u8.shape[1]}x{u8.shape[0]})")


if __name__ == "__main__":
    main()
