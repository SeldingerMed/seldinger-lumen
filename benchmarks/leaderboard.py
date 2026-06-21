"""Navigation benchmark + leaderboard (doc M5).

Runs a policy on a suite of procedural anatomies and emits a JSON leaderboard
(success rate, steps-to-target, peak wall contact). External groups submit by
running their policy through the same NavEnv and reporting this JSON -- the public
benchmark/leaderboard the doc calls the standard-setting public good (doc §7).

A trivial proportional controller is included as the reference baseline. Run:
    python -m benchmarks.leaderboard
"""

from __future__ import annotations

import json

import numpy as np

from lumen.assets import procedural
from lumen.envs import NavEnv


def proportional_policy(obs):
    """Reference baseline: push proportional to remaining signed distance."""
    remaining = obs[4]                      # (target - tip_s)/L
    return np.array([np.clip(4.0 * remaining, -1.0, 1.0)], dtype=np.float32)


def _suite():
    return {
        "straight": procedural.straight_tube(length=80.0, radius=2.0),
        "stenotic": procedural.stenotic_tube(length=80.0, radius=2.0, severity=0.5),
        "narrow": procedural.straight_tube(length=80.0, radius=1.4),
    }


def run_leaderboard(policy=proportional_policy, policy_name="proportional-baseline"):
    results = []
    for name, asset in _suite().items():
        env = NavEnv(asset=asset, target_frac=0.7, max_steps=40)
        obs, _ = env.reset(seed=0)
        peak_contact, done = 0.0, False
        info = {}
        while not done:
            obs, reward, term, trunc, info = env.step(policy(obs))
            peak_contact = max(peak_contact, info["max_r"])
            done = term or trunc
        results.append({"case": name, "success": bool(info["success"]),
                        "steps": env.steps, "final_dist": round(info["dist"], 3),
                        "peak_contact_r": round(peak_contact, 3)})
    n_ok = sum(r["success"] for r in results)
    return {"policy": policy_name, "n_cases": len(results),
            "success_rate": n_ok / len(results), "cases": results}


def main():
    print(json.dumps(run_leaderboard(), indent=2))


if __name__ == "__main__":
    main()
