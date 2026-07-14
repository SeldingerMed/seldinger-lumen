"""Compatibility entry point for the canonical navigation benchmark.

The public benchmark schema lives in :mod:`lumen.bench`; this module remains so
older ``python -m benchmarks.leaderboard`` workflows still work, but it now emits
the same safe-success / max-penetration metrics as the examples and README.

A simple proportional controller is included as a reference policy. Run:
    python -m benchmarks.leaderboard
"""

from __future__ import annotations

import json

import numpy as np

from lumen.bench import evaluate_policy, validate_scorecard


def proportional_policy(obs):
    """Reference baseline: push proportional to remaining signed distance."""
    remaining = obs[4]                      # (target - tip_s)/L
    insertion = np.clip(4.0 * remaining, -1.0, 1.0)
    return np.array([insertion, 0.0], dtype=np.float32)


def run_leaderboard(policy=proportional_policy, policy_name="proportional-baseline"):
    """Evaluate a policy on ``lumen.bench.SUITE`` and return JSON-safe metrics.

    ``cases`` is kept as a backwards-compatible alias for the task rows, but the
    fields are canonical benchmark fields rather than the older peak-contact summary.
    """
    card = validate_scorecard(evaluate_policy(policy, policy_name))
    tasks = [
        {"case": t["name"], "tier": t["tier"], "episodes": t["episodes"],
         "success_rate": t["success_rate"], "safe_success_rate": t["safe_success_rate"],
         "unsafe_success_rate": t.get("unsafe_success_rate", 0.0),
         "mean_steps": t["mean_steps"], "max_pen": t["max_pen"],
         "mean_return": t["mean_return"]}
        for t in card.per_task
    ]
    return {"policy": card.name, "suite_version": card.suite_version,
            "n_cases": len(tasks),
            "success_rate": card.overall["success_rate"],
            "safe_success_rate": card.overall["safe_success_rate"],
            "unsafe_success_rate": card.overall.get("unsafe_success_rate", 0.0),
            "max_pen": card.overall["max_pen"],
            "mean_return": card.overall["mean_return"],
            "cases": tasks}


def main():
    print(json.dumps(run_leaderboard(), indent=2))


if __name__ == "__main__":
    main()
