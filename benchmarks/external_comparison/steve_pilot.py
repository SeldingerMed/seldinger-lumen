"""Run stEVE_bench pilot policies inside the stEVE/SOFA Docker image."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.external_comparison.common_bench import EpisodeResult, _write_results  # noqa: E402


def patch_vtk_for_arm64() -> None:
    """PyVista imports vtkCapsuleSource, missing from the available VTK arm64 wheel."""
    try:
        import vtkmodules.vtkFiltersSources as sources

        if not hasattr(sources, "vtkCapsuleSource") and hasattr(sources, "vtkSphereSource"):
            sources.vtkCapsuleSource = sources.vtkSphereSource
    except Exception:
        pass


def make_policy_action(policy: str, action_space: Any, rng: np.random.Generator, step_idx: int) -> np.ndarray:
    if policy == "random":
        return np.asarray(action_space.sample(), dtype=np.float32)
    action = np.zeros(action_space.shape, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    if policy == "forward":
        action[:, 0] = np.minimum(high[:, 0], 25.0)
    elif policy == "sweep":
        action[:, 0] = np.minimum(high[:, 0], 25.0)
        action[:, 1] = high[:, 1] * np.sin(0.23 * step_idx)
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policies", default="random,forward,sweep")
    parser.add_argument("--tasks", default="BasicWireNav,ArchVariety,DualDeviceNav")
    parser.add_argument("--out-dir", default="benchmarks/external_comparison/results")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    patch_vtk_for_arm64()
    sys.path.insert(0, "/opt/eve_training/training_scripts")

    from eve_bench import ArchVariety, BasicWireNav, DualDeviceNav
    from util.env import BenchEnv

    task_map = {
        "BasicWireNav": (BasicWireNav, "branch_or_arch_navigation"),
        "ArchVariety": (ArchVariety, "branch_or_arch_navigation"),
        "DualDeviceNav": (DualDeviceNav, "dual_device_or_advanced_intervention"),
    }
    selected_tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    policies = [name.strip() for name in args.policies.split(",") if name.strip()]
    episodes: list[EpisodeResult] = []
    for task_name in selected_tasks:
        cls, task_class = task_map[task_name]
        for policy in policies:
            intervention = cls()
            env = BenchEnv(intervention=intervention, mode="eval", visualisation=False, n_max_steps=args.max_steps)
            try:
                for ep_idx in range(args.episodes):
                    if args.progress:
                        print(
                            f"[steve] task={task_name} policy={policy} episode={ep_idx + 1}/{args.episodes}",
                            file=sys.stderr,
                            flush=True,
                        )
                    seed = args.seed + ep_idx
                    rng = np.random.default_rng(seed)
                    if hasattr(env.action_space, "seed"):
                        env.action_space.seed(seed)
                    start = time.perf_counter()
                    try:
                        obs, info = env.reset(seed=seed)
                        total_reward = 0.0
                        success = False
                        final_info: dict[str, Any] = {}
                        steps = 0
                        for step_idx in range(args.max_steps):
                            action = make_policy_action(policy, env.action_space, rng, step_idx)
                            obs, reward, terminated, truncated, final_info = env.step(action)
                            total_reward += float(reward)
                            steps += 1
                            success = success or bool(final_info.get("success", False))
                            if terminated or truncated:
                                break
                        episodes.append(
                            EpisodeResult(
                                environment="steve",
                                task=task_name,
                                task_class=task_class,
                                policy=policy,
                                seed=seed,
                                success=success,
                                safe_success=success,
                                steps=steps,
                                total_reward=total_reward,
                                final_distance=None,
                                elapsed_sec=time.perf_counter() - start,
                                notes={
                                    "path_ratio": final_info.get("path_ratio"),
                                    "average_translation_speed": final_info.get("average translation speed"),
                                    "trajectory_length": final_info.get("trajectory length"),
                                },
                            )
                        )
                    except Exception as exc:
                        episodes.append(
                            EpisodeResult(
                                environment="steve",
                                task=task_name,
                                task_class=task_class,
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
            finally:
                env.close()
    _write_results(
        Path(args.out_dir),
        args.run_id or f"steve-pilot-{int(time.time())}",
        "steve",
        [{"name": name, "task_class": task_map[name][1]} for name in selected_tasks],
        episodes,
        extra={"docker_image": "lumen-steve-cpu:20260715", "vtk_patch": "vtkCapsuleSource->vtkSphereSource"},
    )


if __name__ == "__main__":
    main()
