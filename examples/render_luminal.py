"""L1.4 — render the second sensor modality (luminal RGB) and the fluoro realism seam.

    python examples/render_luminal.py [out_prefix]

Writes <prefix>_luminal.png (endoscopic RGB tunnel view from the device tip) and
<prefix>_fluoro_noisy.png (the same scene as a realistic, dose-limited DRR). Pure
numpy + stdlib PNG writer — no matplotlib/PIL.
"""

from __future__ import annotations

import struct
import sys
import zlib

import numpy as np

from lumen.assets import procedural
from lumen.core.frame import CenterlineFrame
from lumen.sensors import FluoroSensor, LuminalCamera, RealismParams


def _png(path, arr_u8):
    """arr_u8 is (H,W) gray or (H,W,3) RGB uint8."""
    if arr_u8.ndim == 2:
        h, w = arr_u8.shape; color = 0; row = lambda r: arr_u8[r].tobytes()
    else:
        h, w = arr_u8.shape[:2]; color = 2; row = lambda r: arr_u8[r].tobytes()
    raw = b"".join(b"\x00" + row(r) for r in range(h))
    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main(prefix="out"):
    asset = procedural.stenotic_tube(80.0, 4.0, severity=0.6)   # a narrowing to see ahead
    pts, lumen = asset.edge_arrays(asset.edges[0])
    pts = np.asarray(pts)
    frame = CenterlineFrame(pts)
    device = np.stack([pts[1], pts[4]])                          # short device at the inlet, forward

    rgb = LuminalCamera(nu=128, nv=128).render(frame, lumen, device)
    _png(f"{prefix}_luminal.png", (255 * rgb).astype(np.uint8))

    sensor = FluoroSensor(res=48, nu=128, nv=128)
    realism = RealismParams(i0=3e3, psf_sigma=1.0, scatter_frac=0.15, beam_hardening=0.05, seed=0)
    A, _ = sensor.render(pts, realism=realism)                   # realistic dose-limited DRR
    g = A - A.min(); g = (255 * g / (g.max() + 1e-9)).astype(np.uint8)
    _png(f"{prefix}_fluoro_noisy.png", g)
    print(f"wrote {prefix}_luminal.png (RGB tunnel) and {prefix}_fluoro_noisy.png (noisy DRR)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
