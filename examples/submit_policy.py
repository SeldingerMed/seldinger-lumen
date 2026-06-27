"""Evaluate a policy and save a benchmark scorecard.

    python examples/submit_policy.py [results_dir] [submission_name]

Copy this file, replace ``policy`` with your controller, and keep the scorecard JSON
it writes. The scorecard is validated before saving so submission mistakes fail early.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

from lumen.bench import evaluate_policy, leaderboard, scorecard_rejections, validate_scorecard


def policy(obs):
    """Example controller: move forward when target remains distal, otherwise ease off.

    ``obs`` follows NavEnv's 5-D convention:
      [tip_s/L, tip_r/R, sin(theta), cos(theta), (target_s - tip_s)/L]
    """
    remaining = float(np.asarray(obs).reshape(-1)[4])
    return np.array([np.clip(4.0 * remaining, -0.2, 1.0)], dtype=np.float32)


def _safe_submission_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    base = os.path.basename(name.strip())
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("._-")
    if not base:
        raise ValueError("name must contain at least one file-safe character")
    return base


def main(results_dir="bench_results", name="example-policy"):
    os.makedirs(results_dir, exist_ok=True)
    safe_name = _safe_submission_name(name)
    scorecard = validate_scorecard(evaluate_policy(policy, safe_name))
    path = os.path.abspath(os.path.join(results_dir, f"{safe_name}.json"))
    root = os.path.abspath(results_dir)
    if os.path.commonpath([root, path]) != root:
        raise ValueError("submission path must stay under results_dir")
    scorecard.save(path)
    print(f"wrote {path}", flush=True)
    print("leaderboard:", flush=True)
    for rank, card in enumerate(leaderboard(results_dir), 1):
        print(f"  {rank}. {card.name:24} safe={card.overall['safe_success_rate']:.2f}  "
              f"unsafe={card.overall.get('unsafe_success_rate', 0.0):.2f}  "
              f"success={card.overall['success_rate']:.2f}  "
              f"max_pen={card.overall['max_pen']:.3f}  "
              f"return={card.overall['mean_return']:.1f}", flush=True)
    skipped = scorecard_rejections(results_dir)
    if skipped:
        print("\nskipped scorecards:", flush=True)
        for item in skipped:
            print(f"  {item['path']}: {item['error']}", flush=True)
    return path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench_results",
         sys.argv[2] if len(sys.argv) > 2 else "example-policy")
