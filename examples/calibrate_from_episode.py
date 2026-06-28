"""L2.3 — sim2sim wall-stiffness calibration from a captured episode.

    python examples/calibrate_from_episode.py

Generates a wall-probe episode at a known stiffness, saves it, reloads it, and runs
the device-as-sensor inverse to recover the stiffness from the stored fluoro frames —
reporting the recovery error against the ground truth in meta.notes.

Noise-free recovery is trivial (the deterministic render makes loss(true)=0 exactly),
so this probes IDENTIFIABILITY honestly with a little detector noise: a mono
out-of-plane view blows up, biplanar holds (the §3.6 gate). The math is numpy; needs
warp/newton importable.
"""

from __future__ import annotations

from lumen.cli import calibrate_main


def main():
    calibrate_main([])


if __name__ == "__main__":
    main()
