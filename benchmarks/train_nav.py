"""Train a navigation policy (CEM) and score it on the leaderboard vs the baseline.

    python -m benchmarks.train_nav            # train + print comparison
    python -m benchmarks.train_nav --json out.json

Closes the learning loop (doc M5): a gradient-free CEM policy, trained across the
benchmark suite with domain randomisation (so it generalises, not overfits one
anatomy), evaluated on the same NavEnv leaderboard as the proportional baseline.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from lumen.assets import procedural
from lumen.rl.cem import make_policy, train_cem
from benchmarks.leaderboard import proportional_policy, run_leaderboard


def _anat(asset):
    pts, lumen = asset.edge_arrays(asset.edges[0])
    return np.asarray(pts), float(np.asarray(lumen.R).mean()), lumen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--iters", type=int, default=22)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    anatomies = [_anat(procedural.straight_tube(80.0, 2.0)),
                 _anat(procedural.stenotic_tube(80.0, 2.0, severity=0.5)),
                 _anat(procedural.straight_tube(80.0, 1.4))]
    print(f"training CEM (pop={args.pop}, iters={args.iters}) across {len(anatomies)} anatomies...")
    theta, hist = train_cem(anatomies=anatomies, pop=args.pop, iters=args.iters,
                            device=args.device,
                            log=lambda r: print(f"  iter {r['iter']:2d}  mean_ret={r['mean_return']:+.2f}"
                                                f"  success={r['success_rate']:.2f}"))
    base = run_leaderboard(proportional_policy, "proportional-baseline")
    cem = run_leaderboard(make_policy(theta), "cem-linear")
    out = {"baseline": base, "trained": cem, "theta": [float(x) for x in theta]}
    print(json.dumps({k: out[k] for k in ("baseline", "trained")}, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
