"""Layer-0 navigation benchmark + leaderboard (doc M5: "external groups can run and submit").

A FIXED suite of canonical procedural tasks (tiered easy→hard), a standard evaluation
protocol, and a portable scorecard so independent policies are comparable on identical
scenes. The same scene factories back the gymnasium registration (`lumen.envs.registration`),
so a benchmark task and a `gymnasium.make("Lumen/...")` env are the identical scene.

A policy is any callable ``obs -> action`` (e.g. `lumen.rl.make_policy(theta)` from a CEM
run, or the `forward_policy` baseline here). Evaluation is sim-only — no real data, no
gymnasium dependency — so anyone can reproduce a number and submit a scorecard.

Metrics per task (over a fixed set of seeded episodes):
  * ``success_rate``  — fraction of episodes whose tip reaches the target band.
  * ``mean_steps``    — mean steps on the successful episodes (efficiency; lower is better).
  * ``max_pen``       — worst wall over-penetration seen (safety; lower is better).
  * ``mean_return``   — mean episode reward.
The overall score is the macro-average success_rate across tasks (tie-broken by safety).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from lumen.envs.registration import make_nav_stenotic, make_nav_tube, make_tree_nav

SUITE_VERSION = "lumen-bench/0"


def forward_policy(obs):
    """Baseline: advance the proximal end at full rate. Solves the suite but inefficiently
    (more steps / more contact on the harder tiers) — the bar a trained policy must beat."""
    return np.array([1.0], dtype=np.float32)


@dataclass
class BenchTask:
    name: str
    tier: str                       # "easy" | "medium" | "hard"
    make_env: object                # () -> env (callable; NavEnv / TreeNavEnv)
    episodes: int = 5
    seed: int = 0


# the canonical suite (fixed scenes + seeds = reproducible across submitters)
SUITE = [
    BenchTask("nav_tube", "easy", lambda: make_nav_tube(max_steps=40), episodes=5, seed=0),
    BenchTask("nav_stenotic", "medium",
              lambda: make_nav_stenotic(severity=0.5, max_steps=40), episodes=5, seed=100),
    BenchTask("nav_tree_branch", "hard",
              lambda: make_tree_nav(target_node="left_out", max_steps=60), episodes=5, seed=200),
]


def run_episode(env, policy, seed) -> dict:
    """Roll one episode to termination/truncation. Returns success / steps / max wall
    over-penetration / total reward. Robust to a diverged env (info carries finite values)."""
    obs, _ = env.reset(seed=seed)
    total_r, max_pen, success, steps = 0.0, 0.0, False, 0
    R = float(getattr(env, "R", 0.0))
    while True:
        obs, r, terminated, truncated, info = env.step(policy(obs))
        total_r += float(r)
        steps += 1
        max_pen = max(max_pen, max(0.0, float(info.get("max_r", 0.0)) - R))
        success = success or bool(info.get("success", False))
        if terminated or truncated:
            break
    return {"success": success, "steps": steps, "max_pen": max_pen, "return": total_r}


def evaluate_task(task: BenchTask, policy) -> dict:
    """Run a task's seeded episodes and aggregate the per-task metrics."""
    env = task.make_env()
    eps = [run_episode(env, policy, seed=task.seed + i) for i in range(task.episodes)]
    won = [e for e in eps if e["success"]]
    return {
        "name": task.name, "tier": task.tier, "episodes": task.episodes,
        "success_rate": len(won) / len(eps),
        "mean_steps": (float(np.mean([e["steps"] for e in won])) if won else None),
        "max_pen": max(e["max_pen"] for e in eps),
        "mean_return": float(np.mean([e["return"] for e in eps])),
    }


@dataclass
class Scorecard:
    """A portable benchmark result (one submission). Mirrors the asset/episode schema's
    dataclass+JSON I/O so it round-trips through a plain directory."""
    name: str                       # submission name (policy / team)
    suite_version: str
    per_task: list                  # list[dict] from evaluate_task
    overall: dict
    provenance: str = "procedural"
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Scorecard":
        with open(path) as f:
            return cls(**json.load(f))


def evaluate_policy(policy, name: str, suite=SUITE, notes=None) -> Scorecard:
    """Evaluate `policy` over the whole suite and return a Scorecard. The overall score is
    the macro-average success_rate (every tier weighted equally); `max_pen` is the worst
    safety violation across all tasks."""
    per = [evaluate_task(t, policy) for t in suite]
    overall = {
        "success_rate": float(np.mean([t["success_rate"] for t in per])),
        "max_pen": max(t["max_pen"] for t in per),
        "mean_return": float(np.mean([t["mean_return"] for t in per])),
    }
    return Scorecard(name=name, suite_version=SUITE_VERSION, per_task=per, overall=overall,
                     notes=notes or {})


def leaderboard(results_dir: str) -> list[Scorecard]:
    """Read every `*.json` scorecard under `results_dir` and rank them: highest overall
    success_rate first, ties broken by the smaller (safer) max_pen. Scorecards from a
    different suite version are skipped (not comparable)."""
    cards = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            c = Scorecard.load(p)
        except (json.JSONDecodeError, TypeError):
            continue
        if c.suite_version == SUITE_VERSION:
            cards.append(c)
    return sorted(cards, key=lambda c: (-c.overall["success_rate"], c.overall["max_pen"]))
