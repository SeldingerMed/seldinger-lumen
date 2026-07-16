"""Train and evaluate PPO/SAC policies for the common endovascular benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

from benchmarks.external_comparison.common_bench import (
    EpisodeResult,
    _aggregate,
    _as_float,
    _git_commit,
    _host_snapshot,
    _write_results,
    SAFETY_FORCE_THRESHOLD,
)


ALGOS = {"ppo": PPO, "sac": SAC}


LUMEN_TASKS = {
    "nav_tube": ("Lumen/NavTube-v0", "simple_target_navigation", {"max_steps": 300}),
    "nav_stenotic": ("Lumen/NavStenotic-v0", "tortuous_or_stenotic_navigation", {"max_steps": 300}),
    "nav_tree_branch": ("Lumen/NavTreeBranch-v0", "branch_or_arch_navigation", {"max_steps": 300}),
    "nav_tortuous": ("Lumen/NavTortuous-v0", "tortuous_or_stenotic_navigation", {"max_steps": 300}),
    "nav_tortuous_tree": ("Lumen/NavTortuousTree-v0", "branch_or_arch_navigation", {"max_steps": 300}),
}


CATHSIM_TASKS = {
    "phantom3_bca": ("cathsim/CathSim-v0", "branch_or_arch_navigation", {"phantom": "phantom3", "target": "bca"}),
    "phantom3_lcca": ("cathsim/CathSim-v0", "branch_or_arch_navigation", {"phantom": "phantom3", "target": "lcca"}),
}


def make_env(environment: str, task: str, max_steps: int, seed: int) -> gym.Env:
    if environment == "lumen":
        from lumen.envs.registration import register_gym_envs

        register_gym_envs()
        env_id, _, kwargs = LUMEN_TASKS[task]
        kwargs = dict(kwargs)
        kwargs["max_steps"] = max_steps
        env = gym.make(env_id, **kwargs)
    elif environment == "cathsim":
        os.environ.setdefault("MUJOCO_GL", "disable")
        import cathsim.gym.envs  # noqa: F401

        env_id, _, kwargs = CATHSIM_TASKS[task]
        kwargs = dict(kwargs)
        env = gym.make(
            env_id,
            dense_reward=True,
            success_reward=10.0,
            delta=0.004,
            use_pixels=False,
            use_segment=False,
            image_size=64,
            return_info=True,
            use_force=True,
            **kwargs,
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    else:
        raise ValueError(f"unknown environment {environment!r}")
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return Monitor(env)


def task_class(environment: str, task: str) -> str:
    return (LUMEN_TASKS if environment == "lumen" else CATHSIM_TASKS)[task][1]


def policy_name(environment: str) -> str:
    return "MultiInputPolicy" if environment == "cathsim" else "MlpPolicy"


def evaluate_model(args: argparse.Namespace, model: Any) -> list[EpisodeResult]:
    episodes: list[EpisodeResult] = []
    for ep_idx in range(args.eval_episodes):
        seed = args.seed + 10000 + ep_idx
        env = make_env(args.environment, args.task, args.max_steps, seed)
        start = time.perf_counter()
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        success = False
        max_pen = 0.0
        final_distance = None
        forces: list[float] = []
        try:
            for _ in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                total_reward += float(reward)
                if args.environment == "lumen":
                    max_pen = max(max_pen, float(info.get("max_pen", 0.0)))
                    final_distance = _as_float(info.get("dist"))
                    success = success or bool(info.get("success", False))
                else:
                    force = _as_float(info.get("forces"))
                    if force is not None:
                        forces.append(force)
                    if "head_pos" in info and "target_pos" in info:
                        final_distance = float(
                            np.linalg.norm(np.asarray(info["head_pos"]) - np.asarray(info["target_pos"]))
                        )
                        success = success or final_distance <= 0.004
                    success = success or bool(terminated and final_distance is not None and final_distance <= 0.004)
                if terminated or truncated:
                    break
            if args.environment == "lumen":
                unsafe = max_pen > 0.3
                episodes.append(
                    EpisodeResult(
                        environment="lumen",
                        task=args.task,
                        task_class=task_class(args.environment, args.task),
                        policy=f"{args.algo}_trained",
                        seed=seed,
                        success=success,
                        safe_success=bool(success and not unsafe),
                        steps=steps,
                        total_reward=total_reward,
                        final_distance=final_distance,
                        max_penetration=max_pen,
                        unsafe_event=unsafe,
                        elapsed_sec=time.perf_counter() - start,
                        notes={"train_steps": args.timesteps},
                    )
                )
            else:
                max_force = max(forces) if forces else None
                mean_force = float(np.mean(forces)) if forces else None
                unsafe = bool(max_force is not None and max_force > SAFETY_FORCE_THRESHOLD)
                episodes.append(
                    EpisodeResult(
                        environment="cathsim",
                        task=args.task,
                        task_class=task_class(args.environment, args.task),
                        policy=f"{args.algo}_trained",
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
                        notes={"train_steps": args.timesteps},
                    )
                )
        except Exception as exc:
            episodes.append(
                EpisodeResult(
                    environment=args.environment,
                    task=args.task,
                    task_class=task_class(args.environment, args.task),
                    policy=f"{args.algo}_trained",
                    seed=seed,
                    success=False,
                    safe_success=False,
                    steps=steps,
                    total_reward=total_reward,
                    final_distance=final_distance,
                    crashed=True,
                    elapsed_sec=time.perf_counter() - start,
                    notes={"exception": repr(exc), "train_steps": args.timesteps},
                )
            )
        finally:
            env.close()
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["lumen", "cathsim"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--algo", choices=sorted(ALGOS), required=True)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--ppo-n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--out-dir", default="benchmarks/external_comparison/results")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-dir", default="benchmarks/external_comparison/models")
    args = parser.parse_args()

    env = make_env(args.environment, args.task, args.max_steps, args.seed)
    cls = ALGOS[args.algo]
    model_kwargs = {
        "seed": args.seed,
        "verbose": args.verbose,
        "policy_kwargs": {"net_arch": [64, 64]},
        "batch_size": args.batch_size,
    }
    if args.algo == "sac":
        model_kwargs["learning_starts"] = min(100, max(1, args.timesteps // 10))
    else:
        model_kwargs["n_steps"] = args.ppo_n_steps
    model = cls(policy_name(args.environment), env, **model_kwargs)
    started = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    train_elapsed = time.perf_counter() - started
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"{args.environment}-{args.task}-{args.algo}-{args.timesteps}-{int(time.time())}"
    model_path = model_dir / f"{run_id}.zip"
    model.save(model_path)
    env.close()
    episodes = evaluate_model(args, model)
    out_dir = Path(args.out_dir)
    _write_results(
        out_dir,
        run_id,
        args.environment,
        [{"name": args.task, "task_class": task_class(args.environment, args.task)}],
        episodes,
        extra={
            "algo": args.algo,
            "timesteps": args.timesteps,
            "seed": args.seed,
            "model_path": str(model_path),
            "train_elapsed_sec": train_elapsed,
            "aggregate": _aggregate(episodes),
            "host": _host_snapshot(),
            "repo_commit": _git_commit(Path(__file__).resolve().parents[2]),
        },
    )
    print(json.dumps({"model": str(model_path), "train_elapsed_sec": train_elapsed}, indent=2))


if __name__ == "__main__":
    main()
