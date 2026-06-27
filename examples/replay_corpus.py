"""L2.2 — load a corpus of case bundles, summarize it, and replay one.

    python examples/replay_corpus.py [episodes_dir]

Run examples/capture_episode.py first to produce replayable bundle directories.
Pure numpy — this reads the schema, it does not run the sim, so it needs no
newton/warp.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lumen.data import CaseBundle, EpisodeDataset, annotation_coverage, replay, summarize


def _clinical_flags(ep):
    metrics = ep.outcome.metrics if isinstance(ep.outcome.metrics, dict) else {}
    tip = metrics.get("tip_target") if isinstance(metrics.get("tip_target"), dict) else {}
    wall = metrics.get("wall_safety") if isinstance(metrics.get("wall_safety"), dict) else {}
    branch = metrics.get("branch_choice") if isinstance(metrics.get("branch_choice"), dict) else {}
    parts = []
    if "success" in tip:
        parts.append(f"tip_target={tip['success']!s}")
    if "perforation_risk" in wall:
        parts.append(f"wall_risk={wall['perforation_risk']!s}")
    if branch.get("correct") is not None:
        parts.append(f"branch={branch['correct']!s}")
    return "  ".join(parts)


def _annotation_flags(ep):
    cov = annotation_coverage(ep)
    parts = [f"{name}={count}/{cov['steps']}"
             for name, count in sorted(cov["sidecars"].items())]
    keypoint_parts = [
        f"{name}={cov['keypoints_present'].get(name, 0)}/{total}"
        for name, total in sorted(cov["keypoints_total"].items())
    ]
    if keypoint_parts:
        parts.append("keypoints(" + " ".join(keypoint_parts) + ")")
    return "  ".join(parts) if parts else "annotations=none"


def main(root="episodes"):
    if not Path(root).is_dir():
        print(f"no episodes under {root!r} — run examples/capture_episode.py first")
        return
    ds = EpisodeDataset(root, validate_on_load=False)
    if len(ds) == 0:
        print(f"no episodes under {root!r} — run examples/capture_episode.py first")
        return
    bundles = []
    skipped = []
    for d in ds.dirs:
        try:
            bundles.append(CaseBundle.load(d))
        except KeyError as e:
            skipped.append((d, f"manifest missing required key {e!s}"))
        except Exception as e:
            skipped.append((d, f"{type(e).__name__}: {e}"))
    if not bundles:
        print(f"no valid case bundles under {root!r}")
        for path, err in skipped:
            print(f"  skipped {path}: {err}")
        return
    print(f"corpus: {summarize([b.episode for b in bundles])}\n")
    for bundle in bundles:
        ep = bundle.episode
        first_obs = next((obs for *_, obs in replay(ep) if obs is not None), None)  # lazy
        shape = None if first_obs is None else first_obs.shape
        print(f"{ep.outcome.label:18s}  steps={ep.outcome.steps:2d}  "
              f"success={ep.outcome.success!s:5s}  final_dist={ep.outcome.final_dist:6.2f}  "
              f"obs{shape}  calib={bundle.calibration.get('type')}  "
              f"{_clinical_flags(ep)}  {_annotation_flags(ep)}  @ {ep.root}")
    if skipped:
        print("\nskipped invalid bundles:")
        for path, err in skipped:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "episodes")
