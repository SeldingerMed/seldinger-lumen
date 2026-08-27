"""Scaled procedural train/held-out evaluation and generalization gaps."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from lumen.bench_stats import summarize_metrics, validate_statistics

from lumen.bench import BenchTask, evaluate_task


SCALED_SUITE_VERSION = "lumen-bench/heldout/1"
DEFAULT_EPISODES_PER_TASK = 100
SUMMARY_METRICS = (
    "success_rate",
    "safe_success_rate",
    "unsafe_success_rate",
    "crash_rate",
    "mean_return",
    "max_pen",
)
RATE_METRICS = {"success_rate", "safe_success_rate", "unsafe_success_rate", "crash_rate"}


@dataclass(frozen=True)
class SplitTask:
    """One deterministic procedural case with an explicit split and seed block."""

    name: str
    split: str
    case_id: str
    tier: str
    make_env: Callable[[], object]
    episodes: int
    seed: int

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "split": self.split,
            "case_id": self.case_id,
            "tier": self.tier,
            "episodes": self.episodes,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ScaledSuite:
    """Disjoint train and held-out procedural cases."""

    train: tuple[SplitTask, ...]
    heldout: tuple[SplitTask, ...]
    version: str = SCALED_SUITE_VERSION

    def __post_init__(self):
        if self.version != SCALED_SUITE_VERSION:
            raise ValueError(f"version must be {SCALED_SUITE_VERSION!r}")
        if not self.train or not self.heldout:
            raise ValueError("scaled suite must contain non-empty train and heldout splits")
        tasks = self.train + self.heldout
        if any(not isinstance(task, SplitTask) for task in tasks):
            raise ValueError("scaled suite tasks must be SplitTask instances")
        if any(task.split != "train" for task in self.train):
            raise ValueError("train tasks must use the 'train' split")
        if any(task.split != "heldout" for task in self.heldout):
            raise ValueError("held-out tasks must use the 'heldout' split")
        if any(task.split not in {"train", "heldout"} for task in tasks):
            raise ValueError("scaled task split must be 'train' or 'heldout'")
        case_ids = [task.case_id for task in tasks]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("train and held-out case_ids must be disjoint")
        if any(task.episodes <= 0 for task in tasks):
            raise ValueError("scaled task episodes must be positive")

    def manifest(self) -> list[dict]:
        return [task.manifest() for task in self.train + self.heldout]


def _make_tube(length: float, radius: float, max_steps: int):
    from lumen.assets import procedural
    from lumen.envs.nav_gym import NavEnv

    return NavEnv(
        asset=procedural.straight_tube(length=length, radius=radius),
        max_steps=max_steps,
    )


def _make_stenotic(length: float, radius: float, severity: float, max_steps: int):
    from lumen.assets import procedural
    from lumen.envs.nav_gym import NavEnv

    return NavEnv(
        asset=procedural.stenotic_tube(length=length, radius=radius, severity=severity),
        max_steps=max_steps,
    )


def _make_tree(radius: float, angle_deg: float, target_node: str, max_steps: int):
    from lumen.assets import procedural
    from lumen.envs.tree_nav import TreeNavEnv

    return TreeNavEnv(
        procedural.bifurcation(radius=radius, angle_deg=angle_deg),
        target_node=target_node,
        max_steps=max_steps,
    )


def _task(name, split, case_id, tier, factory, episodes, seed):
    return SplitTask(name, split, case_id, tier, factory, episodes, seed)


def make_scaled_suite(episodes_per_task: int = DEFAULT_EPISODES_PER_TASK) -> ScaledSuite:
    """Build a fixed-size, disjoint procedural train/held-out suite.

    Geometry, case IDs, and seed blocks are frozen by this function. Increasing the
    episode count scales evaluation without changing the split or leaking held-out
    cases into training.
    """
    if isinstance(episodes_per_task, (bool, np.bool_)) or not isinstance(
        episodes_per_task, (int, np.integer)
    ):
        raise ValueError("episodes_per_task must be a positive integer")
    episodes = int(episodes_per_task)
    if episodes <= 0:
        raise ValueError("episodes_per_task must be a positive integer")
    return ScaledSuite(
        train=(
            _task(
                "train_tube_nominal", "train", "train-tube-nominal", "easy",
                lambda: _make_tube(80.0, 2.0, 80), episodes, 10_000,
            ),
            _task(
                "train_stenotic_moderate", "train", "train-stenosis-moderate", "medium",
                lambda: _make_stenotic(80.0, 2.0, 0.4, 80), episodes, 11_000,
            ),
            _task(
                "train_tree_left", "train", "train-tree-left-35deg", "hard",
                lambda: _make_tree(2.0, 35.0, "left_out", 100), episodes, 12_000,
            ),
        ),
        heldout=(
            _task(
                "heldout_tube_wide", "heldout", "heldout-tube-wide-110", "easy",
                lambda: _make_tube(110.0, 2.6, 100), episodes, 20_000,
            ),
            _task(
                "heldout_stenotic_severe", "heldout", "heldout-stenosis-severe", "medium",
                lambda: _make_stenotic(90.0, 2.3, 0.7, 100), episodes, 21_000,
            ),
            _task(
                "heldout_tree_right_55deg", "heldout", "heldout-tree-right-55deg", "hard",
                lambda: _make_tree(2.2, 55.0, "right_out", 120), episodes, 22_000,
            ),
        ),
    )


def _evaluate_split(tasks: tuple[SplitTask, ...], policy) -> dict:
    if not tasks:
        raise ValueError("split must contain at least one task")
    rows = []
    episode_metrics = {metric: [] for metric in SUMMARY_METRICS}
    have_episode_metrics = True
    for spec in tasks:
        result = evaluate_task(
            BenchTask(spec.name, spec.tier, spec.make_env, episodes=spec.episodes, seed=spec.seed),
            policy,
        )
        values = result.pop("_episode_metrics", None)
        if not isinstance(values, dict) or any(metric not in values for metric in SUMMARY_METRICS):
            have_episode_metrics = False
        else:
            for metric in SUMMARY_METRICS:
                episode_metrics[metric].extend(values[metric])
        rows.append({**result, "split": spec.split, "case_id": spec.case_id, "seed": spec.seed})
    summary = {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in SUMMARY_METRICS
    }
    statistics = summarize_metrics(episode_metrics, seed=min(task.seed for task in tasks))
    if not have_episode_metrics:
        statistics = {}
    return {"tasks": rows, **summary, "statistics": statistics}


@dataclass
class GeneralizationReport:
    """Portable result with per-split summaries and explicit train-minus-held-out gaps."""

    name: str
    suite_version: str
    manifest: list[dict]
    train: dict
    heldout: dict
    generalization_gap: dict
    provenance: str = "procedural"
    statistics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GeneralizationReport":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return validate_report(cls(**payload))


def validate_report(report: GeneralizationReport) -> GeneralizationReport:
    """Validate split disjointness, aggregate metrics, and gap arithmetic."""
    errors = []
    if not isinstance(report.name, str) or not report.name.strip():
        errors.append("name must be a non-empty string")
    if report.suite_version != SCALED_SUITE_VERSION:
        errors.append(f"suite_version must be {SCALED_SUITE_VERSION!r}")
    if report.provenance != "procedural":
        errors.append("provenance must be 'procedural'")
    statistics = report.statistics
    if statistics:
        if not isinstance(statistics, dict):
            errors.append("statistics must be a mapping")
        else:
            for split_name in ("train", "heldout"):
                split_statistics = statistics.get(split_name)
                if not split_statistics:
                    errors.append(f"statistics.{split_name} must be present")
                    continue
                try:
                    validate_statistics(split_statistics, expected_metrics=SUMMARY_METRICS)
                except ValueError as exc:
                    errors.append(f"statistics.{split_name}: {exc}")

    manifest = report.manifest
    manifest_by_case = {}
    split_case_ids = {"train": set(), "heldout": set()}
    if not isinstance(manifest, list) or not manifest:
        errors.append("manifest must be a non-empty list")
        manifest = []
    for index, item in enumerate(manifest):
        if not isinstance(item, dict):
            errors.append(f"manifest[{index}] must be a mapping")
            continue
        case_id = item.get("case_id")
        split = item.get("split")
        if not isinstance(case_id, str) or not case_id:
            errors.append("manifest case_id values must be non-empty strings")
        elif case_id in manifest_by_case:
            errors.append("manifest case_ids must be unique across splits")
        else:
            manifest_by_case[case_id] = item
        if split not in split_case_ids:
            errors.append("manifest split values must be 'train' or 'heldout'")
        elif isinstance(case_id, str) and case_id:
            split_case_ids[split].add(case_id)
    if not split_case_ids["train"] or not split_case_ids["heldout"]:
        errors.append("manifest must contain both train and heldout cases")
    if split_case_ids["train"] & split_case_ids["heldout"]:
        errors.append("train and held-out case_ids must be disjoint")

    split_payloads = {}
    for split_name, payload in (("train", report.train), ("heldout", report.heldout)):
        split_payloads[split_name] = payload
        if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
            errors.append(f"{split_name} must contain a task list")
            continue
        tasks = payload["tasks"]
        if not tasks:
            errors.append(f"{split_name} task list must not be empty")
            continue
        summary_valid = True
        for metric in SUMMARY_METRICS:
            value = payload.get(metric)
            if not _finite(value):
                errors.append(f"{split_name}.{metric} must be finite")
                summary_valid = False
            elif metric in RATE_METRICS and not 0.0 <= float(value) <= 1.0:
                errors.append(f"{split_name}.{metric} must be in [0, 1]")
                summary_valid = False
        task_values = {metric: [] for metric in SUMMARY_METRICS}
        task_case_ids = []
        for index, task in enumerate(tasks):
            if not isinstance(task, dict):
                errors.append(f"{split_name} tasks must be mappings")
                summary_valid = False
                continue
            if task.get("split") != split_name:
                errors.append(f"{split_name} task split labels must match their container")
            case_id = task.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                errors.append(f"{split_name} task {index} case_id must be a non-empty string")
            elif case_id in task_case_ids:
                errors.append(f"{split_name} task case_ids must be unique")
            else:
                task_case_ids.append(case_id)
            manifest_item = (
                manifest_by_case.get(case_id) if isinstance(case_id, str) else None
            )
            if manifest_item is None or manifest_item.get("split") != split_name:
                errors.append(f"{split_name} task {index} must reference its manifest case")
            elif any(task.get(key) != manifest_item.get(key)
                     for key in ("name", "tier", "episodes", "seed")):
                errors.append(f"{split_name} task {index} metadata must match its manifest case")
            for metric in SUMMARY_METRICS:
                value = task.get(metric)
                if not _finite(value):
                    errors.append(f"{split_name} task {metric} must be finite")
                    summary_valid = False
                else:
                    if metric in RATE_METRICS and not 0.0 <= float(value) <= 1.0:
                        errors.append(f"{split_name} task {metric} must be in [0, 1]")
                        summary_valid = False
                    task_values[metric].append(float(value))
        if set(task_case_ids) != split_case_ids[split_name]:
            errors.append(f"{split_name} task case_ids must equal its manifest cases")
        if summary_valid:
            for metric in SUMMARY_METRICS:
                expected = float(np.mean(task_values[metric]))
                if not np.isclose(float(payload[metric]), expected, atol=1e-9, rtol=0.0):
                    errors.append(f"{split_name}.{metric} must equal the task mean")

    gap = report.generalization_gap
    if not isinstance(gap, dict):
        errors.append("generalization_gap must be a mapping")
    elif all(isinstance(split_payloads.get(split), dict) for split in ("train", "heldout")):
        for metric in SUMMARY_METRICS:
            train_value = split_payloads["train"].get(metric)
            heldout_value = split_payloads["heldout"].get(metric)
            expected = float(train_value) - float(heldout_value) if (
                _finite(train_value) and _finite(heldout_value)
            ) else np.nan
            if not _finite(gap.get(metric)) or not np.isclose(
                float(gap[metric]), expected, atol=1e-9, rtol=0.0
            ):
                errors.append(f"generalization_gap.{metric} must equal train minus heldout")
    if errors:
        raise ValueError("invalid generalization report: " + "; ".join(errors))
    return report


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def evaluate_generalization(
    policy,
    name: str = "policy",
    *,
    episodes_per_task: int = DEFAULT_EPISODES_PER_TASK,
    suite: ScaledSuite | None = None,
) -> GeneralizationReport:
    """Evaluate a policy on disjoint train and held-out cases and report gaps."""
    suite = suite or make_scaled_suite(episodes_per_task)
    train = _evaluate_split(suite.train, policy)
    heldout = _evaluate_split(suite.heldout, policy)
    gap = {
        metric: train[metric] - heldout[metric]
        for metric in SUMMARY_METRICS
    }
    statistics = {}
    if train.get("statistics") and heldout.get("statistics"):
        statistics = {"train": train["statistics"], "heldout": heldout["statistics"]}
    return validate_report(
        GeneralizationReport(
            name=name,
            suite_version=suite.version,
            manifest=suite.manifest(),
            train=train,
            heldout=heldout,
            generalization_gap=gap,
            statistics=statistics,
        )
    )
