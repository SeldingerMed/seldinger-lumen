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
  * ``safe_success_rate`` — fraction that reach the target without a wall-safety breach.
  * ``mean_steps``    — mean steps on the successful episodes (efficiency; lower is better).
  * ``max_pen``       — worst wall over-penetration seen (safety; lower is better).
  * ``mean_return``   — mean episode reward.
Each raw episode result also carries ``clinical`` metrics from
``lumen.data.compute_clinical_metrics``: tip-target success, branch choice, wall
safety, clot/flow fields when present, and catheter support when coaxial.
The leaderboard ranks safe target success first, then raw target success, then wall
safety, then return as a deterministic efficiency tie-break.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from lumen.data import Episode, EpisodeMeta, Outcome, Step, compute_clinical_metrics
from lumen.envs.registration import make_nav_stenotic, make_nav_tube, make_tree_nav

SUITE_VERSION = "lumen-bench/1"
SAFETY_MAX_PEN = 0.3


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
    """Roll one episode to termination/truncation and return generic + clinical metrics."""
    obs, _ = env.reset(seed=seed)
    total_r, max_pen, success, steps = 0.0, 0.0, False, 0
    R = float(getattr(env, "R", 0.0))
    safety_max_pen = float(getattr(env, "safety_max_pen", SAFETY_MAX_PEN))
    trace = []
    target_edge = None
    if getattr(env, "tree", None) is not None and getattr(env, "route", None):
        target_edge = env.tree.edges[env.route[-1]].id
    while True:
        action = policy(obs)
        obs, r, terminated, truncated, info = env.step(action)
        total_r += float(r)
        steps += 1
        if "max_pen" in info:
            max_pen = max(max_pen, float(info["max_pen"]))
        else:
            max_pen = max(max_pen, max(0.0, float(info.get("max_r", 0.0)) - R))
        success = success or bool(info.get("success", False))
        kin = {
            "tip_s": float(info.get("route_s", info.get("tip_s", 0.0))),
            "max_penetration": float(info.get("max_pen", max(0.0, float(info.get("max_r", 0.0)) - R))),
        }
        if info.get("edge") is not None:
            kin["edge"] = info["edge"]
        trace.append(Step(t=float(steps), action={"policy_action": np.asarray(action).reshape(-1).tolist()},
                          kinematics=kin, obs_modality="none"))
        if terminated or truncated:
            break
    notes = {"target_s": float(getattr(env, "target_s", 0.0)),
             "success_tol": float(getattr(env, "success_tol", 2.5)),
             "perforation_penetration_threshold": safety_max_pen}
    labels = {"target_edge": target_edge} if target_edge else {}
    ep = Episode(meta=EpisodeMeta(labels=labels, notes=notes), steps=trace,
                 outcome=Outcome(success=success, final_dist=float(info.get("dist", 0.0)),
                                 steps=steps))
    clinical = compute_clinical_metrics(ep)
    safe_success = bool(success and not clinical["wall_safety"]["perforation_risk"])
    return {"success": success, "safe_success": safe_success, "steps": steps,
            "max_pen": max_pen, "return": total_r, "clinical": clinical}


def evaluate_task(task: BenchTask, policy) -> dict:
    """Run a task's seeded episodes and aggregate the per-task metrics."""
    env = task.make_env()
    safety_max_pen = float(getattr(env, "safety_max_pen", SAFETY_MAX_PEN))
    eps = [run_episode(env, policy, seed=task.seed + i) for i in range(task.episodes)]
    won = [e for e in eps if e["success"]]
    safe_won = [e for e in eps if e["safe_success"]]
    return {
        "name": task.name, "tier": task.tier, "episodes": task.episodes,
        "success_rate": len(won) / len(eps),
        "safe_success_rate": len(safe_won) / len(eps),
        "mean_steps": (float(np.mean([e["steps"] for e in won])) if won else None),
        "max_pen": max(e["max_pen"] for e in eps),
        "safety_max_pen": safety_max_pen,
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


def _finite_number(x) -> bool:
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _rate_ok(x) -> bool:
    return _finite_number(x) and 0.0 <= float(x) <= 1.0


def _close(a, b, tol=1e-9) -> bool:
    return _finite_number(a) and _finite_number(b) and abs(float(a) - float(b)) <= tol


def validate_scorecard(card: Scorecard, suite=SUITE) -> Scorecard:
    """Validate a benchmark submission before it enters a leaderboard.

    Raises ``ValueError`` with actionable schema/comparability errors. Returns the
    original card on success so callers can write ``validate_scorecard(Scorecard.load(p))``.
    """
    errors = []
    if not isinstance(card.name, str) or not card.name.strip():
        errors.append("name must be a non-empty string")
    if card.provenance != "procedural":
        errors.append(f"provenance must be 'procedural', got {card.provenance!r}")
    if card.suite_version != SUITE_VERSION:
        errors.append(f"suite_version must be {SUITE_VERSION!r}, got {card.suite_version!r}")
    expected_names = [t.name for t in suite]
    task_names = ([t.get("name") if isinstance(t, dict) else None for t in card.per_task]
                  if isinstance(card.per_task, list) else [])
    if task_names != expected_names:
        errors.append(f"per_task names must be {expected_names}, got {task_names}")

    if not isinstance(card.overall, dict):
        errors.append("overall must be a dict")
    else:
        for key in ("success_rate", "safe_success_rate"):
            if not _rate_ok(card.overall.get(key)):
                errors.append(f"overall.{key} must be a finite rate in [0, 1]")
        for key in ("max_pen", "mean_return"):
            if not _finite_number(card.overall.get(key)):
                errors.append(f"overall.{key} must be finite")

    if isinstance(card.per_task, list):
        for i, task in enumerate(card.per_task):
            if not isinstance(task, dict):
                errors.append(f"per_task[{i}] must be a dict")
                continue
            if i < len(suite):
                expected = suite[i]
                if task.get("tier") != expected.tier:
                    errors.append(f"per_task[{i}].tier must be {expected.tier!r}")
                if task.get("episodes") != expected.episodes:
                    errors.append(f"per_task[{i}].episodes must be {expected.episodes}")
            for key in ("success_rate", "safe_success_rate"):
                if not _rate_ok(task.get(key)):
                    errors.append(f"per_task[{i}].{key} must be a finite rate in [0, 1]")
            for key in ("episodes", "max_pen", "mean_return"):
                if not _finite_number(task.get(key)):
                    errors.append(f"per_task[{i}].{key} must be finite")
        if (len(card.per_task) == len(suite)
                and isinstance(card.overall, dict)
                and all(isinstance(t, dict) for t in card.per_task)):
            expected_success = float(np.mean([float(t.get("success_rate", np.nan))
                                              for t in card.per_task]))
            expected_safe = float(np.mean([float(t.get("safe_success_rate", np.nan))
                                           for t in card.per_task]))
            expected_max_pen = max(float(t.get("max_pen", np.nan)) for t in card.per_task)
            expected_return = float(np.mean([float(t.get("mean_return", np.nan))
                                             for t in card.per_task]))
            expected = {
                "success_rate": expected_success,
                "safe_success_rate": expected_safe,
                "max_pen": expected_max_pen,
                "mean_return": expected_return,
            }
            for key, value in expected.items():
                if not _close(card.overall.get(key), value):
                    errors.append(f"overall.{key} must equal aggregate per_task {value:.12g}")
    else:
        errors.append("per_task must be a list")

    if errors:
        raise ValueError("invalid benchmark scorecard: " + "; ".join(errors))
    return card


def evaluate_policy(policy, name: str, suite=SUITE, notes=None) -> Scorecard:
    """Evaluate `policy` over the whole suite and return a Scorecard.

    ``success_rate`` is raw target reach. ``safe_success_rate`` is the clinical
    leaderboard metric: target reach with wall penetration below the safety limit.
    Every tier is weighted equally; ``max_pen`` is the worst violation across tasks.
    """
    per = [evaluate_task(t, policy) for t in suite]
    overall = {
        "success_rate": float(np.mean([t["success_rate"] for t in per])),
        "safe_success_rate": float(np.mean([t["safe_success_rate"] for t in per])),
        "max_pen": max(t["max_pen"] for t in per),
        "mean_return": float(np.mean([t["mean_return"] for t in per])),
    }
    return Scorecard(name=name, suite_version=SUITE_VERSION, per_task=per, overall=overall,
                     notes=notes or {})


def _safe_success_for_ranking(card: Scorecard) -> float:
    if "safe_success_rate" in card.overall:
        return float(card.overall["safe_success_rate"])
    # Backward-compatible reading of older scorecards from the same suite: if they
    # predate the explicit field, conservatively zero out unsafe target hits.
    success = float(card.overall.get("success_rate", 0.0))
    return success if float(card.overall.get("max_pen", 0.0)) < SAFETY_MAX_PEN else 0.0


def _load_scorecards(results_dir: str):
    cards, rejected = [], []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            c = Scorecard.load(p)
        except (json.JSONDecodeError, TypeError) as e:
            rejected.append({"path": p, "error": f"could not read scorecard: {e}"})
            continue
        if c.suite_version != SUITE_VERSION:
            rejected.append({"path": p, "error": f"suite_version {c.suite_version!r} "
                             f"is not comparable with {SUITE_VERSION!r}"})
            continue
        try:
            cards.append(validate_scorecard(c))
        except ValueError as e:
            rejected.append({"path": p, "error": str(e)})
    return cards, rejected


def scorecard_rejections(results_dir: str) -> list[dict]:
    """Return scorecard files skipped by the current-suite leaderboard and why."""
    _, rejected = _load_scorecards(results_dir)
    return rejected


def leaderboard(results_dir: str) -> list[Scorecard]:
    """Read every `*.json` scorecard under `results_dir` and rank them.

    Ranking is clinical-first: safe target success, then raw target success, then the
    smaller (safer) max penetration, then higher mean return. Scorecards from other
    suite versions are skipped.
    """
    cards, _ = _load_scorecards(results_dir)
    return sorted(cards, key=lambda c: (-_safe_success_for_ranking(c),
                                       -c.overall["success_rate"],
                                       c.overall["max_pen"],
                                       -c.overall["mean_return"],
                                       c.name))
