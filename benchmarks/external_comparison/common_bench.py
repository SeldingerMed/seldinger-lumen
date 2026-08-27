"""Common endovascular benchmark harness for Lumen and external comparator smoke checks.

This module intentionally keeps environment imports inside runner functions so the same
file can be executed from Lumen's environment or from a comparator-specific virtualenv.
It writes raw per-episode JSON plus an aggregate CSV/JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


COMPARATOR_ENV = "cath" + "sim"
COMPARATOR_GYM_ID = COMPARATOR_ENV + "/" + "Cath" + "Sim-v0"


@dataclass
class EpisodeResult:
    environment: str
    task: str
    task_class: str
    policy: str
    seed: int
    success: bool
    steps: int
    total_reward: float
    final_distance: float | None
    native_safety_pass: bool | None = None
    safety_endpoint: str = "unavailable"
    safety_value: float | None = None
    safety_curve: list[float] | None = None
    wall_load_max: float | None = None
    wall_load_curve: list[float] | None = None
    wall_pressure_curve: list[float] | None = None
    wall_impulse_curve: list[float] | None = None
    wall_load_impulse: float | None = None
    max_penetration: float | None = None
    max_contact_force: float | None = None
    mean_contact_force: float | None = None
    unsafe_event: bool | None = None
    crashed: bool = False
    elapsed_sec: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def steps_per_second(self) -> float:
        if self.elapsed_sec <= 0:
            return 0.0
        return float(self.steps) / float(self.elapsed_sec)


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _host_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
    }


def _policy_action(policy: str, action_space: Any, rng: np.random.Generator, step_idx: int) -> np.ndarray:
    if policy == "random":
        if action_space is not None and hasattr(action_space, "sample"):
            sample = action_space.sample()
            return np.asarray(sample, dtype=np.float32)
        return rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
    if policy == "forward":
        return np.asarray([1.0, 0.0], dtype=np.float32)
    if policy == "sweep":
        return np.asarray([1.0, math.sin(0.23 * step_idx)], dtype=np.float32)
    raise ValueError(f"unknown policy {policy!r}")


def _selected_specs(task_specs: list[dict[str, Any]], task_filter: str) -> list[dict[str, Any]]:
    if not task_filter:
        return task_specs
    selected = {name.strip() for name in task_filter.split(",") if name.strip()}
    return [spec for spec in task_specs if spec["name"] in selected]


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
        if arr.size == 0:
            return None
        return float(np.linalg.norm(arr)) if arr.size > 1 else float(arr.reshape(-1)[0])
    except Exception:
        return None


def _write_results(
    out_dir: Path,
    run_id: str,
    environment: str,
    task_specs: list[dict[str, Any]],
    episodes: list[EpisodeResult],
    extra: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in episodes:
        row = asdict(ep)
        row["steps_per_second"] = ep.steps_per_second
        rows.append(row)
    payload = {
        "run_id": run_id,
        "created_unix": time.time(),
        "environment": environment,
        "task_specs": task_specs,
        "host": _host_snapshot(),
        "extra": extra or {},
        "episodes": rows,
        "aggregate": _aggregate(episodes),
    }
    json_path = out_dir / f"{run_id}.json"
    csv_path = out_dir / f"{run_id}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else list(asdict(EpisodeResult(
            environment="", task="", task_class="", policy="", seed=0, success=False,
            steps=0, total_reward=0.0, final_distance=None,
        )).keys())
        if "steps_per_second" not in fieldnames:
            fieldnames.append("steps_per_second")
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "aggregate": payload["aggregate"]}, indent=2))


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None



def _statistics_protocol(values: dict[str, list[float]], seed: int) -> dict:
    """Use the shared IQM/bootstrap protocol without requiring package installation."""
    try:
        from lumen.bench_stats import summarize_metrics
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "lumen_bench_stats_fallback", root / "lumen" / "bench_stats.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load the shared statistics protocol")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        summarize_metrics = module.summarize_metrics
    return summarize_metrics(values, seed=seed)

def _aggregate(episodes: list[EpisodeResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[EpisodeResult]] = {}
    for ep in episodes:
        grouped.setdefault((ep.environment, ep.task, ep.policy), []).append(ep)
    rows = []
    for (env, task, policy), eps in sorted(grouped.items()):
        successes = [ep for ep in eps if ep.success]
        endpoints = {ep.safety_endpoint for ep in eps}
        single_endpoint = len(endpoints) == 1
        native_pass = [ep.native_safety_pass for ep in eps
                       if single_endpoint and ep.native_safety_pass is not None]
        native_unsafe = [ep.unsafe_event for ep in eps
                         if single_endpoint and ep.unsafe_event is not None]
        safety_values = [ep.safety_value for ep in eps
                         if single_endpoint and ep.safety_value is not None]
        statistics_values = {
            "success_rate": [float(ep.success) for ep in eps],
            "crash_rate": [float(ep.crashed) for ep in eps],
            "mean_return": [ep.total_reward for ep in eps],
            "mean_steps_all": [ep.steps for ep in eps],
            "steps_per_second": [ep.steps_per_second for ep in eps],
        }
        if native_pass:
            statistics_values["native_safety_pass_rate"] = [float(value) for value in native_pass]
        if native_unsafe:
            statistics_values["native_unsafe_event_rate"] = [
                float(value) for value in native_unsafe
            ]
        if safety_values:
            statistics_values["max_safety_value"] = safety_values
        wall_load_values = [ep.wall_load_max for ep in eps if ep.wall_load_max is not None]
        if wall_load_values:
            statistics_values["max_wall_load"] = wall_load_values
        wall_pressure_values = [
            max(ep.wall_pressure_curve) for ep in eps if ep.wall_pressure_curve
        ]
        if wall_pressure_values:
            statistics_values["max_wall_pressure"] = wall_pressure_values
        wall_impulse_values = [
            ep.wall_load_impulse for ep in eps if ep.wall_load_impulse is not None
        ]
        if wall_impulse_values:
            statistics_values["wall_load_impulse"] = wall_impulse_values
        statistics = _statistics_protocol(statistics_values, seed=min(ep.seed for ep in eps))
        rows.append(
            {
                "environment": env,
                "task": task,
                "policy": policy,
                "episodes": len(eps),
                "success_rate": sum(ep.success for ep in eps) / len(eps),
                "native_safety_pass_rate": (
                    sum(native_pass) / len(native_pass) if native_pass else None
                ),
                "safety_endpoint": endpoints.pop() if len(endpoints) == 1 else "mixed",
                "max_safety_value": max(safety_values) if safety_values else None,
                "mean_safety_value": _mean(safety_values),
                "crash_rate": sum(ep.crashed for ep in eps) / len(eps),
                "native_unsafe_event_rate": (
                    sum(native_unsafe) / len(native_unsafe) if native_unsafe else None
                ),
                "mean_steps_success": _mean([ep.steps for ep in successes]),
                "mean_steps_all": _mean([ep.steps for ep in eps]),
                "mean_final_distance": _mean(
                    [ep.final_distance for ep in eps if ep.final_distance is not None]
                ),
                "mean_return": _mean([ep.total_reward for ep in eps]),
                "max_contact_force": _mean(
                    [ep.max_contact_force for ep in eps if ep.max_contact_force is not None]
                ),
                "mean_contact_force": _mean(
                    [ep.mean_contact_force for ep in eps if ep.mean_contact_force is not None]
                ),
                "max_wall_load": _mean(
                    [ep.wall_load_max for ep in eps if ep.wall_load_max is not None]
                ),
                "max_wall_pressure": _mean(
                    [max(ep.wall_pressure_curve) for ep in eps if ep.wall_pressure_curve]
                ),
                "wall_load_impulse": _mean(
                    [ep.wall_load_impulse for ep in eps if ep.wall_load_impulse is not None]
                ),
                "steps_per_second": _mean([ep.steps_per_second for ep in eps]),
                "statistics": statistics,
            }
        )
    return rows


def run_lumen(args: argparse.Namespace) -> None:
    from lumen.envs.registration import (
        make_nav_stenotic,
        make_nav_tortuous,
        make_nav_tube,
        make_tortuous_tree_nav,
        make_tree_nav,
    )

    task_specs = [
        {"name": "nav_tube", "task_class": "simple_target_navigation", "factory": lambda: make_nav_tube(max_steps=args.max_steps)},
        {
            "name": "nav_stenotic",
            "task_class": "tortuous_or_stenotic_navigation",
            "factory": lambda: make_nav_stenotic(severity=0.5, max_steps=args.max_steps),
        },
        {
            "name": "nav_tree_branch",
            "task_class": "branch_or_arch_navigation",
            "factory": lambda: make_tree_nav(target_node="left_out", max_steps=args.max_steps),
        },
        {
            "name": "nav_tortuous",
            "task_class": "tortuous_or_stenotic_navigation",
            "factory": lambda: make_nav_tortuous(max_steps=args.max_steps),
        },
        {
            "name": "nav_tortuous_tree",
            "task_class": "branch_or_arch_navigation",
            "factory": lambda: make_tortuous_tree_nav(target_node="right_out", max_steps=args.max_steps),
        },
    ]
    episodes: list[EpisodeResult] = []
    policies = args.policies.split(",")
    task_specs = _selected_specs(task_specs, args.tasks)
    for task in task_specs:
        for policy in policies:
            env = task["factory"]()
            for ep_idx in range(args.episodes):
                _progress(
                    args.progress,
                    f"[lumen] task={task['name']} policy={policy} episode={ep_idx + 1}/{args.episodes}",
                )
                seed = args.seed + ep_idx
                rng = np.random.default_rng(seed)
                if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
                    env.action_space.seed(seed)
                start = time.perf_counter()
                try:
                    obs, _ = env.reset(seed=seed)
                    total_reward = 0.0
                    max_pen = 0.0
                    penetration_curve: list[float] = []
                    wall_load_curve: list[float] = []
                    wall_pressure_curve: list[float] = []
                    wall_impulse_curve: list[float] = []
                    success = False
                    final_distance = None
                    diverged = False
                    steps = 0
                    for step_idx in range(args.max_steps):
                        action = _policy_action(policy, getattr(env, "action_space", None), rng, step_idx)
                        obs, reward, terminated, truncated, info = env.step(action)
                        steps += 1
                        total_reward += float(reward)
                        diverged = diverged or bool(info.get("diverged", False))
                        if "max_pen" in info:
                            max_pen = max(max_pen, float(info["max_pen"]))
                            penetration_curve.append(float(info["max_pen"]))
                        wall_load = _as_float(info.get("wall_load_max"))
                        if wall_load is not None:
                            wall_load_curve.append(wall_load)
                        wall_pressure = _as_float(info.get("wall_pressure_max"))
                        if wall_pressure is not None:
                            wall_pressure_curve.append(wall_pressure)
                        wall_impulse = _as_float(info.get("wall_load_impulse"))
                        if wall_impulse is not None:
                            wall_impulse_curve.append(wall_impulse)
                        final_distance = _as_float(info.get("dist"))
                        success = success or bool(info.get("success", False))
                        if terminated or truncated:
                            break
                    safety_limit = float(getattr(env, "safety_max_pen", 0.3))
                    unsafe = max_pen > safety_limit
                    episode_notes = {
                        "safety_max_pen": safety_limit,
                        "wall_load_units": "sim_units",
                    }
                    if diverged:
                        episode_notes["failure_reason"] = "sim_diverged"
                    episodes.append(
                        EpisodeResult(
                            environment="lumen",
                            task=task["name"],
                            task_class=task["task_class"],
                            policy=policy,
                            seed=seed,
                            success=success,
                            steps=steps,
                            total_reward=total_reward,
                            final_distance=final_distance,
                            native_safety_pass=bool(success and not unsafe and not diverged),
                            safety_endpoint="surface_penetration_sim_units",
                            safety_value=max_pen,
                            safety_curve=penetration_curve,
                            wall_load_max=max(wall_load_curve) if wall_load_curve else None,
                            wall_load_curve=wall_load_curve,
                            wall_pressure_curve=wall_pressure_curve,
                            wall_impulse_curve=wall_impulse_curve,
                            wall_load_impulse=(
                                wall_impulse_curve[-1] if wall_impulse_curve else None
                            ),
                            max_penetration=max_pen,
                            unsafe_event=unsafe,
                            crashed=diverged,
                            elapsed_sec=time.perf_counter() - start,
                            notes=episode_notes,
                        )
                    )
                except Exception as exc:
                    episodes.append(
                        EpisodeResult(
                            environment="lumen",
                            task=task["name"],
                            task_class=task["task_class"],
                            policy=policy,
                            seed=seed,
                            success=False,
                            steps=0,
                            total_reward=0.0,
                            final_distance=None,
                            native_safety_pass=False,
                            safety_endpoint="surface_penetration_sim_units",
                            safety_value=None,
                            crashed=True,
                            elapsed_sec=time.perf_counter() - start,
                            notes={"exception": repr(exc)},
                        )
                    )
    compact_specs = [{k: v for k, v in spec.items() if k != "factory"} for spec in task_specs]
    _write_results(
        Path(args.out_dir),
        args.run_id or f"lumen-pilot-{int(time.time())}",
        "lumen",
        compact_specs,
        episodes,
        extra={"repo_commit": _git_commit(Path(__file__).resolve().parents[2])},
    )


def run_comparator(args: argparse.Namespace) -> None:
    os.environ.setdefault("MUJOCO_GL", "disable")
    importlib.import_module(COMPARATOR_ENV + ".gym.envs")
    import gymnasium as gym

    task_specs = [
        {"name": "phantom3_bca", "task_class": "branch_or_arch_navigation", "target": "bca"},
        {"name": "phantom3_lcca", "task_class": "branch_or_arch_navigation", "target": "lcca"},
    ]
    policies = args.policies.split(",")
    episodes: list[EpisodeResult] = []
    task_specs = _selected_specs(task_specs, args.tasks)
    for task in task_specs:
        for policy in policies:
            env = None
            try:
                env = gym.make(
                    COMPARATOR_GYM_ID,
                    dense_reward=True,
                    success_reward=10.0,
                    delta=args.delta,
                    use_pixels=False,
                    use_segment=False,
                    image_size=64,
                    phantom="phantom3",
                    target=task["target"],
                    return_info=True,
                    use_force=True,
                )
            except Exception as exc:
                for ep_idx in range(args.episodes):
                    episodes.append(
                        EpisodeResult(
                            environment=COMPARATOR_ENV,
                            task=task["name"],
                            task_class=task["task_class"],
                            policy=policy,
                            seed=args.seed + ep_idx,
                            success=False,
                            steps=0,
                            total_reward=0.0,
                            final_distance=None,
                            native_safety_pass=None,
                            safety_endpoint="contact_force_native_units",
                            safety_value=None,
                            crashed=True,
                            notes={"exception": repr(exc), "phase": "make_env"},
                        )
                    )
                continue
            for ep_idx in range(args.episodes):
                _progress(
                    args.progress,
                    f"[{COMPARATOR_ENV}] task={task['name']} policy={policy} episode={ep_idx + 1}/{args.episodes}",
                )
                seed = args.seed + ep_idx
                rng = np.random.default_rng(seed)
                start = time.perf_counter()
                try:
                    if hasattr(env.action_space, "seed"):
                        env.action_space.seed(seed)
                    obs, info = env.reset(seed=seed)
                    total_reward = 0.0
                    forces: list[float] = []
                    final_distance = None
                    success = False
                    steps = 0
                    for step_idx in range(args.max_steps):
                        action = _policy_action(policy, env.action_space, rng, step_idx)
                        obs, reward, terminated, truncated, info = env.step(action)
                        steps += 1
                        total_reward += float(reward)
                        force = _as_float(info.get("forces"))
                        if force is not None:
                            forces.append(force)
                        if "head_pos" in info and "target_pos" in info:
                            final_distance = float(
                                np.linalg.norm(np.asarray(info["head_pos"]) - np.asarray(info["target_pos"]))
                            )
                            success = success or final_distance <= args.delta
                        success = success or bool(terminated)
                        if terminated or truncated:
                            break
                    max_force = max(forces) if forces else None
                    mean_force = float(np.mean(forces)) if forces else None
                    episodes.append(
                        EpisodeResult(
                            environment=COMPARATOR_ENV,
                            task=task["name"],
                            task_class=task["task_class"],
                            policy=policy,
                            seed=seed,
                            success=success,
                            steps=steps,
                            total_reward=total_reward,
                            final_distance=final_distance,
                            native_safety_pass=None,
                            safety_endpoint="contact_force_native_units",
                            safety_value=max_force,
                            safety_curve=forces,
                            max_contact_force=max_force,
                            mean_contact_force=mean_force,
                            unsafe_event=None,
                            elapsed_sec=time.perf_counter() - start,
                            notes={
                                "delta": args.delta,
                                "safety_classification": "unclassified_pending_physical_calibration",
                            },
                        )
                    )
                except Exception as exc:
                    episodes.append(
                        EpisodeResult(
                            environment=COMPARATOR_ENV,
                            task=task["name"],
                            task_class=task["task_class"],
                            policy=policy,
                            seed=seed,
                            success=False,
                            steps=0,
                            total_reward=0.0,
                            final_distance=None,
                            native_safety_pass=None,
                            safety_endpoint="contact_force_native_units",
                            safety_value=None,
                            crashed=True,
                            notes={"exception": repr(exc)},
                        )
                    )
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    _write_results(
        Path(args.out_dir),
        args.run_id or f"{COMPARATOR_ENV}-pilot-{int(time.time())}",
        COMPARATOR_ENV,
        task_specs,
        episodes,
        extra={"repo_commit": _git_commit(Path(args.external_repo).resolve()) if args.external_repo else None},
    )


def smoke_steve(args: argparse.Namespace) -> None:
    checks = {
        "python": sys.version,
        "eve_importable": importlib.util.find_spec("eve") is not None,
        "eve_bench_importable": importlib.util.find_spec("eve_bench") is not None,
        "sofa_importable": importlib.util.find_spec("Sofa") is not None,
        "sofa_runtime_importable": importlib.util.find_spec("SofaRuntime") is not None,
        "sofa_root": os.environ.get("SOFA_ROOT"),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }
    result = {
        "run_id": args.run_id or f"steve-smoke-{int(time.time())}",
        "created_unix": time.time(),
        "host": _host_snapshot(),
        "checks": checks,
        "status": "ready" if all(checks[k] for k in ("eve_importable", "eve_bench_importable", "sofa_importable", "sofa_runtime_importable")) else "blocked",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['run_id']}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"json": str(path), **result}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("lumen", COMPARATOR_ENV):
        p = sub.add_parser(name)
        p.add_argument("--episodes", type=int, default=30)
        p.add_argument("--max-steps", type=int, default=300)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--policies", default="random,forward,sweep")
        p.add_argument("--tasks", default="")
        p.add_argument("--progress", action="store_true")
        p.add_argument("--out-dir", default="benchmarks/external_comparison/results")
        p.add_argument("--run-id", default="")
        if name == COMPARATOR_ENV:
            p.add_argument("--delta", type=float, default=0.004)
            p.add_argument("--external-repo", default="")
        p.set_defaults(func=run_lumen if name == "lumen" else run_comparator)
    p = sub.add_parser("smoke-steve")
    p.add_argument("--out-dir", default="benchmarks/external_comparison/results")
    p.add_argument("--run-id", default="")
    p.set_defaults(func=smoke_steve)
    args = parser.parse_args()
    random.seed(args.seed if hasattr(args, "seed") else 0)
    np.random.seed(args.seed if hasattr(args, "seed") else 0)
    args.func(args)


if __name__ == "__main__":
    main()
