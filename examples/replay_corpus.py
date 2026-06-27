"""L2.2 — load a corpus of case bundles, summarize it, and replay one.

    python examples/replay_corpus.py [episodes_dir]

Run examples/capture_episode.py first to produce replayable bundle directories.
Pure numpy — this reads the schema, it does not run the sim, so it needs no
newton/warp.
"""

from __future__ import annotations

import sys

from lumen.cli import replay_main


def main(root="episodes"):
    replay_main([str(root)])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "episodes")
