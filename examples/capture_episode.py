"""L2.1 — capture procedural episodes into the Layer-2 schema.

    python examples/capture_episode.py [out_dir]

Generates a few procedural cases (straight / stenotic), runs the guidewire to the
target while recording the paired observation each step, and writes one replayable
case-bundle directory per case under <out_dir>. Reloads them and prints a summary.
Needs the full stack (newton + warp).
"""

from __future__ import annotations

import sys

from lumen.workflows import capture_examples, write_preview_sheet

_write_preview_sheet = write_preview_sheet


def main(out_dir="episodes"):
    capture_examples(out_dir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "episodes")
