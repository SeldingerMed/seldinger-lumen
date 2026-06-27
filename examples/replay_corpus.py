"""L2.2 — load a corpus of case bundles, summarize it, and replay one.

    python examples/replay_corpus.py [episodes_dir]

Run examples/capture_episode.py first to produce replayable bundle directories.
Pure numpy — this reads the schema, it does not run the sim, so it needs no
newton/warp.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lumen.data import CaseBundle, EpisodeDataset, replay, summarize


def main(root="episodes"):
    if not Path(root).is_dir():
        print(f"no episodes under {root!r} — run examples/capture_episode.py first")
        return
    ds = EpisodeDataset(root)
    if len(ds) == 0:
        print(f"no episodes under {root!r} — run examples/capture_episode.py first")
        return
    print(f"corpus: {summarize(ds)}\n")
    for ep in ds:
        bundle = CaseBundle.load(ep.root)
        first_obs = next((obs for *_, obs in replay(ep) if obs is not None), None)  # lazy
        shape = None if first_obs is None else first_obs.shape
        print(f"{ep.outcome.label:18s}  steps={ep.outcome.steps:2d}  "
              f"success={ep.outcome.success!s:5s}  final_dist={ep.outcome.final_dist:6.2f}  "
              f"obs{shape}  calib={bundle.calibration.get('type')}  @ {ep.root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "episodes")
