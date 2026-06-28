"""Render a guidewire to synthetic fluoroscopy (Layer 1 L1.0) and save previews.

    python examples/render_fluoro.py [out.png]

Pure numpy + stdlib PNG writer (no matplotlib/PIL). Uses a synthetic curved wire; to
render a live device, pass NewtonGuidewireSim.body_positions() as `nodes`.
"""

from __future__ import annotations

import sys

from lumen.workflows import render_fluoro_example


def main():
    render_fluoro_example(sys.argv[1] if len(sys.argv) > 1 else "fluoro.png")


if __name__ == "__main__":
    main()
