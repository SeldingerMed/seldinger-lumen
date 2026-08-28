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
  * ``safe_success_rate`` — Lumen-native fraction reaching the target without
    exceeding the device-surface overlap limit in simulator units.
  * ``unsafe_success_rate`` — Lumen-native target reach after exceeding that limit.
  * ``mean_steps``    — mean steps on the successful episodes (efficiency; lower is better).
  * ``max_pen``       — worst device-surface overlap proxy seen (lower is better).
  * ``mean_return``   — mean episode reward.
  * ``crash_rate``    — fraction of episodes ended by a finite divergence guard.
Each raw episode result also carries native wall-load, pressure-proxy, and impulse
curves in simulator units plus ``clinical`` metrics from
``lumen.data.compute_clinical_metrics``. These native safety fields are not
cross-environment calibrated endpoints.
The leaderboard ranks Lumen-native safe target success first, then raw target
success, then wall safety, then return as a deterministic efficiency tie-break.
Generated scorecards carry a SHA-256 replay certificate for every action/outcome;
`replay_verified_leaderboard` re-runs those episodes with the supplied policy before
ranking them. They also carry ``lumen-stats/1`` episode-level IQMs, means, and 95%
percentile bootstrap intervals with the seed and resample count recorded.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from lumen.data import Episode, EpisodeMeta, Outcome, Step, compute_clinical_metrics
from lumen.bench_stats import summarize_metrics, validate_statistics
from lumen.envs.registration import make_nav_stenotic, make_nav_tube, make_tree_nav

SUITE_VERSION = "lumen-bench/3"
REPLAY_PROTOCOL_VERSION = "lumen-bench-replay/1"
SAFETY_MAX_PEN = 0.3
STAT_METRICS = (
    "success_rate",
    "safe_success_rate",
    "unsafe_success_rate",
    "crash_rate",
    "mean_return",
    "max_pen",
)


def forward_policy(obs):
    """Baseline: advance the proximal end at full rate. Solves the suite but inefficiently
    (more steps / more contact on the harder tiers) — the bar a trained policy must beat."""
    return np.array([1.0, 0.0], dtype=np.float32)


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

REPLAY_OUTCOME_KEYS = (
    "success",
    "safe_success",
    "steps",
    "max_pen",
    "return",
    "wall_load_max",
    "wall_pressure_max",
    "wall_load_impulse",
    "crashed",
    "diverged",
)


def _replay_digest(task_name: str, seed: int, actions: list, outcome: dict) -> str:
    payload = {
        "task": task_name,
        "seed": int(seed),
        "actions": actions,
        "outcome": {key: outcome[key] for key in REPLAY_OUTCOME_KEYS},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _replay_outcome(result: dict) -> dict:
    outcome = {}
    for key in REPLAY_OUTCOME_KEYS:
        if key == "wall_pressure_max":
            outcome[key] = result.get(
                key, max(result.get("wall_pressure_curve", ()), default=0.0)
            )
        else:
            outcome[key] = result[key]
    return outcome


def _validate_replay_record(record: object, task: BenchTask, episode_index: int) -> list[str]:
    errors = []
    if not isinstance(record, dict):
        return [f"episode {episode_index} must be a dict"]
    expected_seed = task.seed + episode_index
    if record.get("task") != task.name:
        errors.append(f"episode {episode_index}.task must be {task.name!r}")
    if record.get("seed") != expected_seed:
        errors.append(f"episode {episode_index}.seed must be {expected_seed}")
    actions = record.get("actions")
    if not isinstance(actions, list):
        errors.append(f"episode {episode_index}.actions must be a list")
    else:
        for action_index, action in enumerate(actions):
            if not isinstance(action, list) or not action:
                errors.append(f"episode {episode_index}.actions[{action_index}] must be non-empty")
                continue
            try:
                values = np.asarray(action, dtype=float)
            except (TypeError, ValueError):
                errors.append(f"episode {episode_index}.actions[{action_index}] must be numeric")
                continue
            if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < -1.0) or np.any(values > 1.0):
                errors.append(f"episode {episode_index}.actions[{action_index}] must be finite and in [-1, 1]")
    outcome = record.get("outcome")
    if not isinstance(outcome, dict):
        errors.append(f"episode {episode_index}.outcome must be a dict")
    else:
        for key in REPLAY_OUTCOME_KEYS:
            if key not in outcome:
                errors.append(f"episode {episode_index}.outcome missing {key}")
        if errors:
            return errors
        if not isinstance(outcome["success"], bool) or not isinstance(outcome["safe_success"], bool):
            errors.append(f"episode {episode_index}.outcome success fields must be booleans")
        if not isinstance(outcome["crashed"], bool) or not isinstance(outcome["diverged"], bool):
            errors.append(f"episode {episode_index}.outcome crash fields must be booleans")
        if not isinstance(outcome["steps"], (int, np.integer)) or isinstance(
            outcome["steps"], (bool, np.bool_)
        ) or int(outcome["steps"]) < 0:
            errors.append(f"episode {episode_index}.outcome.steps must be a non-negative integer")
        elif isinstance(actions, list) and int(outcome["steps"]) != len(actions):
            errors.append(f"episode {episode_index}.outcome.steps must equal action count")
        for key in ("max_pen", "return", "wall_load_max",
                    "wall_pressure_max", "wall_load_impulse"):
            try:
                if not np.isfinite(float(outcome[key])):
                    errors.append(f"episode {episode_index}.outcome.{key} must be finite")
            except (TypeError, ValueError):
                errors.append(f"episode {episode_index}.outcome.{key} must be numeric")
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append(f"episode {episode_index}.sha256 must be a 64-character string")
    elif isinstance(actions, list) and isinstance(outcome, dict) and not errors:
        try:
            expected = _replay_digest(task.name, expected_seed, actions, outcome)
        except (TypeError, ValueError):
            errors.append(f"episode {episode_index}.replay payload is not JSON-safe")
        else:
            if digest != expected:
                errors.append(f"episode {episode_index}.sha256 does not match replay payload")
    return errors


def run_episode(env, policy, seed, *, record_replay: bool = False) -> dict:
    """Roll one episode and retain native surface/load/pressure/impulse traces."""
    obs, _ = env.reset(seed=seed)
    total_r, max_pen, success, steps, diverged = 0.0, 0.0, False, 0, False
    wall_load_curve, wall_pressure_curve, wall_impulse_curve = [], [], []
    safety_max_pen = float(getattr(env, "safety_max_pen", SAFETY_MAX_PEN))
    R = float(getattr(env, "R", 0.0))
    trace = []
    replay_actions = []
    target_edge = None
    if getattr(env, "tree", None) is not None and getattr(env, "route", None):
        target_edge = env.tree.edges[env.route[-1]].id
    while True:
        action = policy(obs)
        if record_replay:
            replay_actions.append(np.asarray(action).reshape(-1).tolist())
        obs, r, terminated, truncated, info = env.step(action)
        total_r += float(r)
        diverged = diverged or bool(info.get("diverged", False))
        steps += 1
        if "max_pen" in info:
            max_pen = max(max_pen, float(info["max_pen"]))
        else:
            max_pen = max(max_pen, max(0.0, float(info.get("max_r", 0.0)) - R))
        for key, curve in (
            ("wall_load_max", wall_load_curve),
            ("wall_pressure_max", wall_pressure_curve),
            ("wall_load_impulse", wall_impulse_curve),
        ):
            value = info.get(key)
            if value is not None and np.isfinite(float(value)):
                curve.append(float(value))
        success = success or bool(info.get("success", False))
        kin = {
            "tip_s": float(info.get("route_s", info.get("tip_s", 0.0))),
            "max_penetration": float(
                info.get("max_pen", max(0.0, float(info.get("max_r", 0.0)) - R))
            ),
        }
        if "wall_load_max" in info:
            kin["wall_load_max"] = float(info["wall_load_max"])
        if "wall_pressure_max" in info:
            kin["wall_pressure_max"] = float(info["wall_pressure_max"])
        if "wall_load_impulse" in info:
            kin["wall_load_impulse"] = float(info["wall_load_impulse"])
        if info.get("edge") is not None:
            kin["edge"] = info["edge"]
        trace.append(Step(
            t=float(steps),
            action={"policy_action": np.asarray(action).reshape(-1).tolist()},
            kinematics=kin,
            obs_modality="none",
        ))
        if terminated or truncated:
            break
    notes = {
        "target_s": float(getattr(env, "target_s", 0.0)),
        "success_tol": float(getattr(env, "success_tol", 2.5)),
        "native_safety_endpoint": "surface_penetration_sim_units",
        "perforation_penetration_threshold": safety_max_pen,
        "wall_load_units": "sim_units",
    }
    labels = {"target_edge": target_edge} if target_edge else {}
    ep = Episode(
        meta=EpisodeMeta(labels=labels, notes=notes),
        steps=trace,
        outcome=Outcome(
            success=success, final_dist=float(info.get("dist", 0.0)), steps=steps
        ),
    )
    clinical = compute_clinical_metrics(ep)
    safe_success = bool(success and not diverged and not clinical["wall_safety"]["perforation_risk"])
    result = {
        "success": success,
        "safe_success": safe_success,
        "steps": steps,
        "max_pen": max_pen,
        "return": total_r,
        "wall_load_max": max(wall_load_curve) if wall_load_curve else 0.0,
        "wall_load_curve": wall_load_curve,
        "wall_pressure_curve": wall_pressure_curve,
        "wall_impulse_curve": wall_impulse_curve,
        "wall_load_impulse": wall_impulse_curve[-1] if wall_impulse_curve else 0.0,
        "clinical": clinical,
        "crashed": diverged,
        "diverged": diverged,
    }
    if record_replay:
        result["_replay"] = {
            "seed": int(seed),
            "actions": replay_actions,
            "outcome": _replay_outcome(result),
        }
    return result

def evaluate_task(task: BenchTask, policy) -> dict:
    """Run a task's seeded episodes and aggregate the per-task metrics."""
    env = task.make_env()
    safety_max_pen = float(getattr(env, "safety_max_pen", SAFETY_MAX_PEN))
    eps = [run_episode(env, policy, seed=task.seed + i,
                       record_replay=True) for i in range(task.episodes)]
    replay = []
    for episode in eps:
        record = episode.pop("_replay")
        record["task"] = task.name
        record["sha256"] = _replay_digest(
            task.name, record["seed"], record["actions"], record["outcome"]
        )
        replay.append(record)
    won = [e for e in eps if e["success"]]
    safe_won = [e for e in eps if e["safe_success"]]
    unsafe_won = [e for e in eps if e["success"] and not e["safe_success"]]
    crashed = [e for e in eps if e["crashed"]]
    episode_metrics = {
        "success_rate": [float(e["success"]) for e in eps],
        "safe_success_rate": [float(e["safe_success"]) for e in eps],
        "unsafe_success_rate": [
            float(e["success"] and not e["safe_success"]) for e in eps
        ],
        "crash_rate": [float(e["crashed"]) for e in eps],
        "mean_return": [e["return"] for e in eps],
        "max_pen": [e["max_pen"] for e in eps],
    }
    return {
        "name": task.name, "tier": task.tier, "episodes": task.episodes, "seed": task.seed,
        "success_rate": len(won) / len(eps),
        "safe_success_rate": len(safe_won) / len(eps),
        "unsafe_success_rate": len(unsafe_won) / len(eps),
        "crash_rate": len(crashed) / len(eps),
        "mean_steps": (float(np.mean([e["steps"] for e in won])) if won else None),
        "max_pen": max(e["max_pen"] for e in eps),
        "safety_max_pen": safety_max_pen,
        "mean_return": float(np.mean([e["return"] for e in eps])),
        "max_wall_load": max(e["wall_load_max"] for e in eps),
        "max_wall_pressure": max(
            (max(e["wall_pressure_curve"]) for e in eps if e["wall_pressure_curve"]),
            default=0.0,
        ),
        "wall_load_impulse": float(np.mean([e["wall_load_impulse"] for e in eps])),
        "statistics": summarize_metrics(episode_metrics, seed=task.seed),
        "_episode_metrics": episode_metrics,
        "_replay": replay,
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
    statistics: dict = field(default_factory=dict)
    replay: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Scorecard":
        with open(path) as f:
            payload = json.load(f)
        payload.setdefault("replay", {})
        return cls(**payload)


def _finite_number(x) -> bool:
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _rate_ok(x) -> bool:
    return _finite_number(x) and 0.0 <= float(x) <= 1.0


def _close(a, b, tol=1e-9) -> bool:
    return _finite_number(a) and _finite_number(b) and abs(float(a) - float(b)) <= tol

def _validate_replay_payload(replay: object, suite) -> list[str]:
    if not isinstance(replay, dict):
        return ["replay must be a dict"]
    errors = []
    if replay.get("protocol") != REPLAY_PROTOCOL_VERSION:
        errors.append(f"replay.protocol must be {REPLAY_PROTOCOL_VERSION!r}")
    if replay.get("suite_version") != SUITE_VERSION:
        errors.append(f"replay.suite_version must be {SUITE_VERSION!r}")
    records = replay.get("episodes")
    if not isinstance(records, list):
        return errors + ["replay.episodes must be a list"]
    expected_count = sum(int(task.episodes) for task in suite)
    if len(records) != expected_count:
        errors.append(f"replay.episodes must contain {expected_count} records")
    offset = 0
    for task in suite:
        for episode_index in range(int(task.episodes)):
            if offset >= len(records):
                break
            errors.extend(_validate_replay_record(records[offset], task, episode_index))
            offset += 1
    return errors

def _replay_aggregate_errors(card: Scorecard, replay: dict, suite) -> list[str]:
    records = replay["episodes"]
    task_rows = []
    offset = 0
    errors = []
    for task in suite:
        group = records[offset:offset + int(task.episodes)]
        offset += int(task.episodes)
        outcomes = [record["outcome"] for record in group]
        successes = [bool(item["success"]) for item in outcomes]
        safe_successes = [bool(item["safe_success"]) for item in outcomes]
        task_rows.append({
            "success_rate": float(np.mean(successes)),
            "safe_success_rate": float(np.mean(safe_successes)),
            "unsafe_success_rate": float(np.mean([
                success and not safe
                for success, safe in zip(successes, safe_successes)
            ])),
            "crash_rate": float(np.mean([bool(item["crashed"]) for item in outcomes])),
            "mean_steps": (
                float(np.mean([item["steps"] for item, success in zip(outcomes, successes)
                               if success]))
                if any(successes) else None
            ),
            "max_pen": max(float(item["max_pen"]) for item in outcomes),
            "mean_return": float(np.mean([item["return"] for item in outcomes])),
            "max_wall_load": max(float(item["wall_load_max"]) for item in outcomes),
            "max_wall_pressure": max(float(item["wall_pressure_max"]) for item in outcomes),
            "wall_load_impulse": float(
                np.mean([item["wall_load_impulse"] for item in outcomes])
            ),
        })

    task_keys = (
        "success_rate",
        "safe_success_rate",
        "unsafe_success_rate",
        "crash_rate",
        "mean_steps",
        "max_pen",
        "mean_return",
        "max_wall_load",
        "max_wall_pressure",
        "wall_load_impulse",
    )
    for index, (actual, expected) in enumerate(zip(task_rows, card.per_task)):
        for key in task_keys:
            if actual[key] is None:
                matches = expected.get(key) is None
            else:
                matches = _close(expected.get(key), actual[key])
            if not matches:
                errors.append(f"replay per_task[{index}].{key} does not match verified episodes")

    overall = {
        "success_rate": float(np.mean([row["success_rate"] for row in task_rows])),
        "safe_success_rate": float(np.mean([row["safe_success_rate"] for row in task_rows])),
        "unsafe_success_rate": float(np.mean([
            row["unsafe_success_rate"] for row in task_rows
        ])),
        "crash_rate": float(np.mean([row["crash_rate"] for row in task_rows])),
        "max_pen": max(row["max_pen"] for row in task_rows),
        "mean_return": float(np.mean([row["mean_return"] for row in task_rows])),
        "max_wall_load": max(row["max_wall_load"] for row in task_rows),
        "max_wall_pressure": max(row["max_wall_pressure"] for row in task_rows),
        "wall_load_impulse": float(np.mean([
            row["wall_load_impulse"] for row in task_rows
        ])),
    }
    for key, actual in overall.items():
        if not _close(card.overall.get(key), actual):
            errors.append(f"replay overall.{key} does not match verified episodes")
    return errors


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
    if card.statistics:
        try:
            validate_statistics(card.statistics, expected_metrics=STAT_METRICS)
        except ValueError as exc:
            errors.append(str(exc))
    if card.suite_version != SUITE_VERSION:
        errors.append(f"suite_version must be {SUITE_VERSION!r}, got {card.suite_version!r}")
    expected_names = [t.name for t in suite]
    task_names = ([t.get("name") if isinstance(t, dict) else None for t in card.per_task]
                  if isinstance(card.per_task, list) else [])
    if task_names != expected_names:
        errors.append(f"per_task names must be {expected_names}, got {task_names}")
    if card.replay:
        replay_errors = _validate_replay_payload(card.replay, suite)
        errors.extend(replay_errors)
        if (
            not replay_errors
            and isinstance(card.per_task, list)
            and len(card.per_task) == len(suite)
            and all(isinstance(task, dict) for task in card.per_task)
            and isinstance(card.overall, dict)
        ):
            errors.extend(_replay_aggregate_errors(card, card.replay, suite))

    if not isinstance(card.overall, dict):
        errors.append("overall must be a dict")
    else:
        for key in ("success_rate", "safe_success_rate",
                    "unsafe_success_rate", "crash_rate"):
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
            task_statistics = task.get("statistics")
            if task_statistics:
                try:
                    validate_statistics(task_statistics, expected_metrics=STAT_METRICS)
                except ValueError as exc:
                    errors.append(f"per_task[{i}].{exc}")
            for key in ("success_rate", "safe_success_rate",
                        "unsafe_success_rate", "crash_rate"):
                if not _rate_ok(task.get(key)):
                    errors.append(f"per_task[{i}].{key} must be a finite rate in [0, 1]")
            if (_rate_ok(task.get("success_rate")) and _rate_ok(task.get("safe_success_rate"))
                    and _rate_ok(task.get("unsafe_success_rate"))
                    and not _close(task.get("unsafe_success_rate"),
                                   float(task.get("success_rate"))
                                   - float(task.get("safe_success_rate")))):
                errors.append(f"per_task[{i}].unsafe_success_rate must equal "
                              "success_rate - safe_success_rate")
            for key in ("episodes", "max_pen", "mean_return"):
                if not _finite_number(task.get(key)):
                    errors.append(f"per_task[{i}].{key} must be finite")
        if (card.per_task
                and len(card.per_task) == len(suite)
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
            expected["unsafe_success_rate"] = expected_success - expected_safe
            expected["crash_rate"] = float(np.mean([
                float(t.get("crash_rate", np.nan)) for t in card.per_task
            ]))
            for key, value in expected.items():
                if not _close(card.overall.get(key), value):
                    errors.append(f"overall.{key} must equal aggregate per_task {value:.12g}")
    else:
        errors.append("per_task must be a list")

    if errors:
        raise ValueError("invalid benchmark scorecard: " + "; ".join(errors))
    return card


def evaluate_policy(policy, name: str, suite=SUITE, notes=None) -> Scorecard:
    """Evaluate ``policy`` over the whole suite and return a Scorecard.

    ``safe_success_rate`` is a Lumen-native metric: target reach with the
    device-surface overlap proxy below the configured simulator-unit limit. The
    scorecard also retains native wall-load, pressure-proxy, and impulse curves;
    none is a calibrated SI injury endpoint.
    """
    raw_per = [evaluate_task(t, policy) for t in suite]
    per = []
    replay_records = []
    all_episode_metrics = {metric: [] for metric in STAT_METRICS}
    have_episode_metrics = True
    for task in raw_per:
        episode_metrics = task.get("_episode_metrics")
        if not isinstance(episode_metrics, dict) or any(
            metric not in episode_metrics for metric in STAT_METRICS
        ):
            have_episode_metrics = False
        else:
            for metric in STAT_METRICS:
                all_episode_metrics[metric].extend(episode_metrics[metric])
        replay_records.extend(task.get("_replay", []))
        per.append({
            key: value for key, value in task.items()
            if key not in {"_episode_metrics", "_replay"}
        })
    overall = {
        "success_rate": float(np.mean([t["success_rate"] for t in per])),
        "safe_success_rate": float(np.mean([t["safe_success_rate"] for t in per])),
        "unsafe_success_rate": float(np.mean([t["unsafe_success_rate"] for t in per])),
        "max_pen": max(t["max_pen"] for t in per),
        "mean_return": float(np.mean([t["mean_return"] for t in per])),
        "max_wall_load": max(t["max_wall_load"] for t in per),
        "max_wall_pressure": max(t["max_wall_pressure"] for t in per),
        "wall_load_impulse": float(np.mean([t["wall_load_impulse"] for t in per])),
        "crash_rate": float(np.mean([t["crash_rate"] for t in per])),
    }
    statistics = summarize_metrics(all_episode_metrics, seed=0) if have_episode_metrics else {}
    replay = {
        "protocol": REPLAY_PROTOCOL_VERSION,
        "suite_version": SUITE_VERSION,
        "episodes": replay_records,
    }
    return Scorecard(name=name, suite_version=SUITE_VERSION, per_task=per, overall=overall,
                     notes=notes or {}, statistics=statistics, replay=replay)

def verify_replay(card: Scorecard, policy, suite=SUITE) -> dict:
    """Re-run a scorecard's recorded actions/outcomes and verify every digest."""
    errors = []
    if not isinstance(card, Scorecard):
        errors.append("card must be a Scorecard")
    else:
        try:
            validate_scorecard(card, suite=suite)
        except ValueError as exc:
            errors.append(str(exc))
        replay = card.replay
        if not isinstance(replay, dict) or not replay.get("episodes"):
            errors.append("scorecard has no replay certificate")
    if errors:
        return {
            "protocol": REPLAY_PROTOCOL_VERSION,
            "verified": False,
            "episodes": 0,
            "errors": errors,
        }

    records = card.replay["episodes"]
    checked = 0
    actual_records = []
    for task in suite:
        env = task.make_env()
        for episode_index in range(int(task.episodes)):
            expected = records[checked]
            checked += 1
            seed = int(task.seed) + episode_index
            try:
                actual = run_episode(env, policy, seed=seed, record_replay=True)
                actual_record = actual.pop("_replay")
                actual_record["task"] = task.name
                actual_record["sha256"] = _replay_digest(
                    task.name,
                    actual_record["seed"],
                    actual_record["actions"],
                    actual_record["outcome"],
                )
                actual_records.append(actual_record)
            except Exception as exc:
                errors.append(
                    f"{task.name}[{episode_index}] replay failed: {type(exc).__name__}: {exc}"
                )
                continue
            if actual_record["sha256"] != expected.get("sha256"):
                errors.append(f"{task.name}[{episode_index}] replay digest mismatch")
    if not errors:
        errors.extend(_replay_aggregate_errors(
            card,
            {"episodes": actual_records},
            suite,
        ))
    return {
        "protocol": REPLAY_PROTOCOL_VERSION,
        "verified": not errors,
        "episodes": checked,
        "errors": errors,
    }


def replay_verified_leaderboard(results_dir: str, policies: dict, suite=SUITE) -> list[Scorecard]:
    """Rank only scorecards whose certificates re-run with a supplied policy."""
    cards, _ = _load_scorecards(results_dir, suite=suite)
    verified = []
    for card in cards:
        policy = policies.get(card.name) if isinstance(policies, dict) else None
        if policy is None:
            continue
        if verify_replay(card, policy, suite=suite)["verified"]:
            verified.append(card)
    return sorted(verified, key=lambda c: (-_safe_success_for_ranking(c),
                                           -c.overall["success_rate"],
                                           c.overall["max_pen"],
                                           -c.overall["mean_return"],
                                           c.name))


def _safe_success_for_ranking(card: Scorecard) -> float:
    if "safe_success_rate" in card.overall:
        return float(card.overall["safe_success_rate"])
    # Backward-compatible reading of older scorecards from the same suite: if they
    # predate the explicit field, conservatively zero out unsafe target hits.
    success = float(card.overall.get("success_rate", 0.0))
    return success if float(card.overall.get("max_pen", 0.0)) < SAFETY_MAX_PEN else 0.0


def _load_scorecards(results_dir: str, suite=SUITE):
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
            cards.append(validate_scorecard(c, suite=suite))
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
