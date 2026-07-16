"""Common endovascular benchmark harness for Lumen, CathSim, and stEVE smoke checks.

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


SAFETY_FORCE_THRESHOLD = 2.0


@dataclass
class EpisodeResult:
    environment: str
    task: str
    task_class: str
    policy: str
    seed: int
    success: bool
    safe_success: bool
    steps: int
    total_reward: float
    final_distance: float | None
    max_penetration: float | None = None
    max_contact_force: float | None = None
    mean_contact_force: float | None = None
    unsafe_event: bool = False
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
        fieldnames = list(rows[0].keys()) if rows else list(asdict(EpisodeResult("", "", "", "", 0, False, False, 0, 0, None)).keys())
        if "steps_per_second" not in fieldnames:
            fieldnames.append("steps_per_second")
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "aggregate": payload["aggregate"]}, indent=2))


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _aggregate(episodes: list[EpisodeResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[EpisodeResult]] = {}
    for ep in episodes:
        grouped.setdefault((ep.environment, ep.task, ep.policy), []).append(ep)
    rows = []
    for (env, task, policy), eps in sorted(grouped.items()):
        successes = [ep for ep in eps if ep.success]
        rows.append(
            {
                "environment": env,
                "task": task,
                "policy": policy,
                "episodes": len(eps),
                "success_rate": sum(ep.success for ep in eps) / len(eps),
                "safe_success_rate": sum(ep.safe_success for ep in eps) / len(eps),
                "crash_rate": sum(ep.crashed for ep in eps) / len(eps),
                "unsafe_event_rate": sum(ep.unsafe_event for ep in eps) / len(eps),
                "mean_steps_success": _mean([ep.steps for ep in successes]),
                "mean_steps_all": _mean([ep.steps for ep in eps]),
                "mean_final_distance": _mean([ep.final_distance for ep in eps if ep.final_distance is not None]),
                "mean_return": _mean([ep.total_reward for ep in eps]),
                "max_contact_force": _mean([ep.max_contact_force for ep in eps if ep.max_contact_force is not None]),
                "mean_contact_force": _mean([ep.mean_contact_force for ep in eps if ep.mean_contact_force is not None]),
                "steps_per_second": _mean([ep.steps_per_second for ep in eps]),
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
                    final_distance = None
                    success = False
                    steps = 0
                    for step_idx in range(args.max_steps):
                        action = _policy_action(policy, getattr(env, "action_space", None), rng, step_idx)
                        obs, reward, terminated, truncated, info = env.step(action)
                        steps += 1
                        total_reward += float(reward)
                        if "max_pen" in info:
                            max_pen = max(max_pen, float(info["max_pen"]))
                        final_distance = _as_float(info.get("dist"))
                        success = success or bool(info.get("success", False))
                        if terminated or truncated:
                            break
                    safety_limit = float(getattr(env, "safety_max_pen", 0.3))
                    unsafe = max_pen > safety_limit
                    episodes.append(
                        EpisodeResult(
                            environment="lumen",
                            task=task["name"],
                            task_class=task["task_class"],
                            policy=policy,
                            seed=seed,
                            success=success,
                            safe_success=bool(success and not unsafe),
                            steps=steps,
                            total_reward=total_reward,
                            final_distance=final_distance,
                            max_penetration=max_pen,
                            unsafe_event=unsafe,
                            elapsed_sec=time.perf_counter() - start,
                            notes={"safety_max_pen": safety_limit},
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
                            safe_success=False,
                            steps=0,
                            total_reward=0.0,
                            final_distance=None,
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


def run_cathsim(args: argparse.Namespace) -> None:
    os.environ.setdefault("MUJOCO_GL", "disable")
    import cathsim.gym.envs  # noqa: F401
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
                    "cathsim/CathSim-v0",
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
                            environment="cathsim",
                            task=task["name"],
                            task_class=task["task_class"],
                            policy=policy,
                            seed=args.seed + ep_idx,
                            success=False,
                            safe_success=False,
                            steps=0,
                            total_reward=0.0,
                            final_distance=None,
                            crashed=True,
                            notes={"exception": repr(exc), "phase": "make_env"},
                        )
                    )
                continue
            for ep_idx in range(args.episodes):
                _progress(
                    args.progress,
                    f"[cathsim] task={task['name']} policy={policy} episode={ep_idx + 1}/{args.episodes}",
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
                    unsafe = bool(max_force is not None and max_force > SAFETY_FORCE_THRESHOLD)
                    episodes.append(
                        EpisodeResult(
                            environment="cathsim",
                            task=task["name"],
                            task_class=task["task_class"],
                            policy=policy,
                            seed=seed,
                            success=success,
                            safe_success=bool(success and not unsafe),
                            steps=steps,
                            total_reward=total_reward,
                            final_distance=final_distance,
                            max_contact_force=max_force,
                            mean_contact_force=mean_force,
                            unsafe_event=unsafe,
                            elapsed_sec=time.perf_counter() - start,
                            notes={"delta": args.delta, "force_threshold": SAFETY_FORCE_THRESHOLD},
                        )
                    )
                except Exception as exc:
                    episodes.append(
                        EpisodeResult(
                            environment="cathsim",
                            task=task["name"],
                            task_class=task["task_class"],
                            policy=policy,
                            seed=seed,
                            success=False,
                            safe_success=False,
                            steps=0,
                            total_reward=0.0,
                            final_distance=None,
                            crashed=True,
                            elapsed_sec=time.perf_counter() - start,
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
        args.run_id or f"cathsim-pilot-{int(time.time())}",
        "cathsim",
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
    for name in ("lumen", "cathsim"):
        p = sub.add_parser(name)
        p.add_argument("--episodes", type=int, default=30)
        p.add_argument("--max-steps", type=int, default=300)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--policies", default="random,forward,sweep")
        p.add_argument("--tasks", default="")
        p.add_argument("--progress", action="store_true")
        p.add_argument("--out-dir", default="benchmarks/external_comparison/results")
        p.add_argument("--run-id", default="")
        if name == "cathsim":
            p.add_argument("--delta", type=float, default=0.004)
            p.add_argument("--external-repo", default="")
        p.set_defaults(func=run_lumen if name == "lumen" else run_cathsim)
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
