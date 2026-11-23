"""Train and evaluate PPO/SAC policies for the common endovascular benchmark."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.monitor import Monitor
except ImportError:  # optional training dependency; parsing helpers remain importable
    PPO = SAC = Monitor = None

from benchmarks.external_comparison.common_bench import (
    EpisodeResult,
    _aggregate,
    _as_float,
    _git_commit,
    _host_snapshot,
    _write_results,
    sanitize_run_id,
)


ALGOS = {"ppo": PPO, "sac": SAC}
COMPARATOR_ENV = "cath" + "sim"
COMPARATOR_GYM_ID = COMPARATOR_ENV + "/" + "Cath" + "Sim-v0"

PREREGISTERED_SEEDS = tuple(range(6))
PREREGISTERED_EVAL_EPISODES = 100
EVAL_SEED_START = 10_000


def parse_seed_schedule(value: str | None, fallback: int = 0) -> list[int]:
    """Parse a unique non-negative comma-separated seed schedule."""
    if value is None or not value.strip():
        values = [str(fallback)]
    else:
        values = value.split(",")
    try:
        seeds = [int(item.strip()) for item in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("seeds must be comma-separated integers") from exc
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    return seeds


def seed_comparator_rng(env: Any, seed: int) -> None:
    """Seed the comparator's hidden RNGs for one episode.

    At the pinned revision its start-pose sampler draws from Python's global ``random`` and
    its task code also uses NumPy's global RNG, neither of which ``reset(seed=...)`` touches.
    The composer environment's own ``random_state`` is reseeded as well, so a given evaluation
    seed replays the same case in every pass (training eval, contact pass, recorder).
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    inner = getattr(getattr(env, "unwrapped", env), "_env", None)
    state = getattr(inner, "random_state", None) or getattr(inner, "_random_state", None)
    if isinstance(state, np.random.RandomState):
        state.seed(seed % (2**32))


LUMEN_TASKS = {
    "nav_tube": ("Lumen/NavTube-v0", "simple_target_navigation", {"max_steps": 300}),
    "nav_stenotic": ("Lumen/NavStenotic-v0", "tortuous_or_stenotic_navigation", {"max_steps": 300}),
    "nav_tree_branch": ("Lumen/NavTreeBranch-v0", "branch_or_arch_navigation", {"max_steps": 300}),
    "nav_tortuous": ("Lumen/NavTortuous-v0", "tortuous_or_stenotic_navigation", {"max_steps": 300}),
    "nav_tortuous_tree": ("Lumen/NavTortuousTree-v0", "branch_or_arch_navigation", {"max_steps": 300}),
}


COMPARATOR_TASKS = {
    "phantom3_bca": (COMPARATOR_GYM_ID, "branch_or_arch_navigation", {"phantom": "phantom3", "target": "bca"}),
    "phantom3_lcca": (COMPARATOR_GYM_ID, "branch_or_arch_navigation", {"phantom": "phantom3", "target": "lcca"}),
}


def make_env(environment: str, task: str, max_steps: int, seed: int) -> gym.Env:
    if environment == "lumen":
        from lumen.envs.registration import register_gym_envs

        register_gym_envs()
        env_id, _, kwargs = LUMEN_TASKS[task]
        kwargs = dict(kwargs)
        kwargs["max_steps"] = max_steps
        env = gym.make(env_id, **kwargs)
    elif environment == COMPARATOR_ENV:
        os.environ.setdefault("MUJOCO_GL", "disable")
        importlib.import_module(COMPARATOR_ENV + ".gym.envs")

        env_id, _, kwargs = COMPARATOR_TASKS[task]
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
        seed_comparator_rng(env, seed)
    else:
        raise ValueError(f"unknown environment {environment!r}")
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return Monitor(env)


def task_class(environment: str, task: str) -> str:
    return (LUMEN_TASKS if environment == "lumen" else COMPARATOR_TASKS)[task][1]


def policy_name(environment: str) -> str:
    return "MultiInputPolicy" if environment == COMPARATOR_ENV else "MlpPolicy"


def evaluate_model(
    args: argparse.Namespace,
    model: Any,
    *,
    training_seed: int | None = None,
    model_id: str | None = None,
) -> list[EpisodeResult]:
    """Evaluate a trained model on one common frozen evaluation seed schedule."""
    training_seed = args.seed if training_seed is None else training_seed
    episodes: list[EpisodeResult] = []
    for ep_idx in range(args.eval_episodes):
        seed = EVAL_SEED_START + ep_idx
        env = None
        start = time.perf_counter()
        total_reward = 0.0
        steps = 0
        success = False
        max_pen = 0.0
        final_distance = None
        forces: list[float] = []
        penetration_curve: list[float] = []
        wall_load_curve: list[float] = []
        wall_pressure_curve: list[float] = []
        wall_impulse_curve: list[float] = []
        diverged = False
        try:
            env = make_env(args.environment, args.task, args.max_steps, seed)
            if args.environment == COMPARATOR_ENV:
                seed_comparator_rng(env, seed)
            obs, _ = env.reset(seed=seed)
            for _ in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                total_reward += float(reward)
                diverged = diverged or bool(info.get("diverged", False))
                if args.environment == "lumen":
                    pen = float(info.get("max_pen", 0.0))
                    max_pen = max(max_pen, pen)
                    penetration_curve.append(pen)
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
                else:
                    force = _as_float(info.get("forces"))
                    if force is not None:
                        forces.append(force)
                    if "head_pos" in info and "target_pos" in info:
                        final_distance = float(
                            np.linalg.norm(np.asarray(info["head_pos"]) - np.asarray(info["target_pos"]))
                        )
                        success = success or final_distance <= 0.004
                    success = success or bool(
                        terminated and final_distance is not None and final_distance <= 0.004
                    )
                if terminated or truncated:
                    break
            if args.environment == "lumen":
                unsafe = max_pen > 0.3
                episode_notes = {
                    "train_steps": args.timesteps,
                    "safety_max_pen": 0.3,
                    "wall_load_units": "sim_units",
                }
                if diverged:
                    episode_notes["failure_reason"] = "sim_diverged"
                episodes.append(
                    EpisodeResult(
                        environment="lumen",
                        task=args.task,
                        task_class=task_class(args.environment, args.task),
                        policy=f"{args.algo}_trained",
                        seed=seed,
                        success=success,
                        steps=steps,
                        total_reward=total_reward,
                        final_distance=final_distance,
                        training_seed=training_seed,
                        model_id=model_id,
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
            else:
                comparator_notes = {
                    "train_steps": args.timesteps,
                    "safety_classification": "unclassified_pending_physical_calibration",
                }
                if diverged:
                    comparator_notes["failure_reason"] = "sim_diverged"
                max_force = max(forces) if forces else None
                mean_force = float(np.mean(forces)) if forces else None
                episodes.append(
                    EpisodeResult(
                        environment=COMPARATOR_ENV,
                        task=args.task,
                        task_class=task_class(args.environment, args.task),
                        policy=f"{args.algo}_trained",
                        seed=seed,
                        success=success,
                        steps=steps,
                        total_reward=total_reward,
                        final_distance=final_distance,
                        training_seed=training_seed,
                        model_id=model_id,
                        native_safety_pass=None,
                        safety_endpoint="contact_force_native_units",
                        safety_value=max_force,
                        safety_curve=forces,
                        max_contact_force=max_force,
                        mean_contact_force=mean_force,
                        unsafe_event=None,
                        crashed=diverged,
                        elapsed_sec=time.perf_counter() - start,
                        notes=comparator_notes,
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
                    steps=steps,
                    total_reward=total_reward,
                    final_distance=final_distance,
                    training_seed=training_seed,
                    model_id=model_id,
                    native_safety_pass=(
                        False if args.environment == "lumen" else None
                    ),
                    safety_endpoint=(
                        "surface_penetration_sim_units"
                        if args.environment == "lumen" else "contact_force_native_units"
                    ),
                    safety_value=None,
                    crashed=True,
                    elapsed_sec=time.perf_counter() - start,
                    notes={"exception": repr(exc), "train_steps": args.timesteps},
                )
            )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    if episodes and episodes[-1].seed == seed:
                        episodes[-1].notes["cleanup_exception"] = repr(exc)
    return episodes


class SeedRunError(RuntimeError):
    """A per-seed failure with stage and timing retained for unconditional reporting."""

    def __init__(
        self,
        stage: str,
        cause: Exception,
        elapsed_sec: float,
        train_elapsed_sec: float = 0.0,
    ):
        self.stage = stage
        self.cause = cause
        self.elapsed_sec = float(elapsed_sec)
        self.train_elapsed_sec = float(train_elapsed_sec)
        super().__init__(f"{stage} failed: {cause!r}")


def _train_one(args: argparse.Namespace, seed: int, model_path: Path):
    started = time.perf_counter()
    train_started = started
    train_elapsed = 0.0
    env = None
    stage = "environment"
    failed = False
    try:
        env = make_env(args.environment, args.task, args.max_steps, seed)
        stage = "model_init"
        model_kwargs = {
            "seed": seed,
            "verbose": args.verbose,
            "policy_kwargs": {"net_arch": [64, 64]},
            "batch_size": args.batch_size,
        }
        if args.algo == "sac":
            model_kwargs["learning_starts"] = min(100, max(1, args.timesteps // 10))
        else:
            model_kwargs["n_steps"] = args.ppo_n_steps
        model = ALGOS[args.algo](policy_name(args.environment), env, **model_kwargs)
        stage = "train"
        train_started = time.perf_counter()
        callbacks, remaining = None, args.timesteps
        every = int(getattr(args, "checkpoint_every", 0) or 0)
        if every > 0:
            # Interruption tolerance only: a periodic snapshot (policy, optimizer, rollout counters)
            # lets a killed seed resume instead of restarting. A resumed seed is recorded as such.
            from stable_baselines3.common.callbacks import CheckpointCallback

            ckpt_dir = model_path.parent / (model_path.stem + "-ckpt")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _check_checkpoint_identity(ckpt_dir, _checkpoint_identity(args, seed))
            if (ckpt_dir / "completed.json").exists():
                # Snapshots are kept for learning curves, never to resume a finished run: a rerun of a
                # completed seed must not silently continue from its last snapshot.
                raise ValueError(f"{ckpt_dir} belongs to a completed run; a rerun needs a new --run-id "
                                 "or an emptied checkpoint directory")
            # Only snapshots strictly before the budget: SB3's CheckpointCallback fires before the
            # final rollout is consumed by an update, so a snapshot at the full budget is not a
            # finished model. Resuming from it would skip the last update.
            done = sorted((p for p in ckpt_dir.glob("ckpt_*_steps.zip")
                           if int(p.stem.split("_")[-2]) < args.timesteps),
                          key=lambda p: int(p.stem.split("_")[-2]))
            for snap in reversed(done):   # newest readable snapshot; one cut short by a kill is skipped
                try:
                    model = ALGOS[args.algo].load(snap, env=env, seed=seed)
                    # The fresh env sits at its reset state, not the checkpoint's: clearing the pickled
                    # last observation makes SB3's _setup_learn reset and start the segment consistently.
                    model._last_obs = None
                except Exception as exc:  # truncated or corrupt zip
                    print(f"skipping unreadable checkpoint {snap.name}: {exc!r}", file=sys.stderr, flush=True)
                    continue
                remaining = max(0, args.timesteps - int(model.num_timesteps))
                args._resumed_from = int(model.num_timesteps)
                break
            callbacks = CheckpointCallback(save_freq=every, save_path=str(ckpt_dir), name_prefix="ckpt")
        if callbacks is None:
            model.learn(total_timesteps=remaining, progress_bar=False)
        else:
            # With reset_num_timesteps=False, SB3 (2.x _setup_learn) adds the loaded num_timesteps to
            # total_timesteps, so passing the remaining steps trains exactly to the budget.
            model.learn(total_timesteps=remaining, progress_bar=False, callback=callbacks,
                        reset_num_timesteps=not getattr(args, "_resumed_from", 0))
        train_elapsed = time.perf_counter() - train_started
    except Exception as exc:
        failed = True
        if stage == "train":
            train_elapsed = time.perf_counter() - train_started
        raise SeedRunError(
            stage,
            exc,
            time.perf_counter() - started,
            train_elapsed,
        ) from exc
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                if not failed:
                    raise SeedRunError(
                        "environment_close",
                        close_exc,
                        time.perf_counter() - started,
                        train_elapsed,
                    ) from close_exc
    stage = "save"
    try:
        model.save(model_path)
        if int(getattr(args, "checkpoint_every", 0) or 0) > 0:
            ck = model_path.parent / (model_path.stem + "-ckpt")
            (ck / "completed.json").write_text(json.dumps({"timesteps": int(args.timesteps),
                                                           "model": str(model_path)}))
    except Exception as exc:
        raise SeedRunError(
            stage,
            exc,
            time.perf_counter() - started,
            train_elapsed,
        ) from exc
    eval_args = argparse.Namespace(**vars(args))
    eval_args.seed = seed
    stage = "evaluate"
    try:
        episodes = evaluate_model(
            eval_args,
            model,
            training_seed=seed,
            model_id=str(model_path),
        )
    except Exception as exc:
        raise SeedRunError(
            stage,
            exc,
            time.perf_counter() - started,
            train_elapsed,
        ) from exc
    return model_path, train_elapsed, time.perf_counter() - started, episodes


def _checkpoint_identity(args: argparse.Namespace, seed: int) -> dict:
    """What a resumable snapshot must match: the run it continues."""
    return {"environment": args.environment, "task": args.task, "algo": args.algo, "seed": int(seed),
            "timesteps": int(args.timesteps), "ppo_n_steps": int(args.ppo_n_steps),
            "batch_size": int(args.batch_size), "max_steps": int(args.max_steps)}


def _check_checkpoint_identity(ckpt_dir: Path, ident: dict) -> None:
    """Refuse to resume snapshots written by a different configuration under the same run id.

    The identity is stored next to the snapshots on first use; a directory from before this
    record existed is adopted as-is (its snapshots carry the seed and rollout settings SB3 saved)."""
    f = ckpt_dir / "run_identity.json"
    if f.exists():
        prev = json.loads(f.read_text())
        diff = {k: (prev.get(k), v) for k, v in ident.items() if prev.get(k) != v}
        if diff:
            raise ValueError(f"checkpoint dir {ckpt_dir} belongs to a different run configuration: {diff}")
        return
    f.write_text(json.dumps(ident, indent=1))


def _failed_seed_episodes(
    args: argparse.Namespace,
    training_seed: int,
    model_id: str,
    error: Exception,
    *,
    stage: str,
    elapsed_sec: float,
    train_elapsed_sec: float,
) -> list[EpisodeResult]:
    """Represent a failed seed across the complete frozen eval schedule."""
    endpoint = (
        "surface_penetration_sim_units"
        if args.environment == "lumen" else "contact_force_native_units"
    )
    notes = {
        "failure_reason": f"{stage}_failed",
        "failure_stage": stage,
        "failure_elapsed_sec": float(elapsed_sec),
        "train_elapsed_sec": float(train_elapsed_sec),
        "exception": repr(error),
        "train_steps": args.timesteps,
    }
    return [
        EpisodeResult(
            environment=args.environment,
            task=args.task,
            task_class=task_class(args.environment, args.task),
            policy=f"{args.algo}_trained",
            seed=EVAL_SEED_START + ep_idx,
            training_seed=training_seed,
            model_id=model_id,
            success=False,
            steps=0,
            total_reward=0.0,
            final_distance=None,
            native_safety_pass=False if args.environment == "lumen" else None,
            safety_endpoint=endpoint,
            crashed=True,
            notes=notes.copy(),
        )
        for ep_idx in range(args.eval_episodes)
    ]


def _task_spec(args: argparse.Namespace) -> dict[str, str]:
    return {"name": args.task, "task_class": task_class(args.environment, args.task)}




def _run_id(args: argparse.Namespace) -> str:
    raw = args.run_id or (
        f"{args.environment}-{args.task}-{args.algo}-{args.timesteps}-{int(time.time())}"
    )
    return sanitize_run_id(raw)


def _main_payload(model_paths, seeds, elapsed, runs):
    payload = {
        "models": [str(path) for path in model_paths],
        "seeds": seeds,
        "train_elapsed_sec": elapsed,
        "runs": runs,
    }
    if len(model_paths) == 1:
        payload["model"] = str(model_paths[0])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["lumen", COMPARATOR_ENV], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--algo", choices=sorted(ALGOS), required=True)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=PREREGISTERED_EVAL_EPISODES)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--ppo-n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0,
                        help="Fallback seed when --seeds is omitted.")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated unique seeds for independent training runs.")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--out-dir", default="benchmarks/external_comparison/results")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model-dir", default="benchmarks/external_comparison/models")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Save a resumable snapshot every N env steps and resume from the latest one (0 = off).")
    args = parser.parse_args()
    task_map = LUMEN_TASKS if args.environment == "lumen" else COMPARATOR_TASKS
    if args.task not in task_map:
        parser.error(f"--task must be one of: {', '.join(sorted(task_map))}")
    try:
        seeds = parse_seed_schedule(args.seeds, fallback=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if ALGOS[args.algo] is None:
        parser.error("stable-baselines3 is required for training")

    try:
        run_id = _run_id(args)
    except ValueError as exc:
        parser.error(str(exc))
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if int(getattr(args, "checkpoint_every", 0) or 0) > 0:
        # Refused before anything is trained or written: a rerun of a completed checkpointed seed would
        # otherwise replace that run's result files with a failed-seed record.
        for seed in seeds:
            mp = model_dir / (f"{run_id}.zip" if len(seeds) == 1 else f"{run_id}-seed{seed}.zip")
            if (mp.parent / (mp.stem + "-ckpt") / "completed.json").exists():
                parser.error(f"seed {seed} of {run_id} already completed (its checkpoint directory is "
                             "marked complete); a rerun needs a new --run-id or an emptied checkpoint directory")
    model_paths = []
    runs = []
    episodes = []
    for seed in seeds:
        args._resumed_from = 0  # resume offset is per seed; never inherited from a failed seed
        model_path = model_dir / (
            f"{run_id}.zip" if len(seeds) == 1 else f"{run_id}-seed{seed}.zip"
        )
        try:
            saved_path, train_elapsed, elapsed_sec, seed_episodes = _train_one(
                args, seed, model_path
            )
        except SeedRunError as exc:
            failure_stage = exc.stage
            failure_error = exc.cause
            failure_elapsed = exc.elapsed_sec
            train_elapsed = exc.train_elapsed_sec
            saved_path = model_path
            seed_episodes = None
        except Exception as exc:
            failure_stage = "unknown"
            failure_error = exc
            failure_elapsed = 0.0
            train_elapsed = 0.0
            saved_path = model_path
            seed_episodes = None
        else:
            model_paths.append(saved_path)
            runs.append({
                "seed": seed,
                "model_path": str(saved_path),
                "status": "completed",
                "train_elapsed_sec": train_elapsed,
                "elapsed_sec": elapsed_sec,
                "eval_episodes": args.eval_episodes,
                "resumed_from_step": int(getattr(args, "_resumed_from", 0)),
            })
            args._resumed_from = 0
        if seed_episodes is None:
            seed_episodes = _failed_seed_episodes(
                args,
                seed,
                str(saved_path),
                failure_error,
                stage=failure_stage,
                elapsed_sec=failure_elapsed,
                train_elapsed_sec=train_elapsed,
            )
            runs.append({
                "seed": seed,
                "model_path": str(model_path),
                "status": "failed",
                "failure_stage": failure_stage,
                "error": repr(failure_error),
                "train_elapsed_sec": train_elapsed,
                "elapsed_sec": failure_elapsed,
                "eval_episodes": args.eval_episodes,
                "resumed_from_step": int(getattr(args, "_resumed_from", 0)),
            })
        episodes.extend(seed_episodes)
    elapsed = float(sum(run["train_elapsed_sec"] for run in runs))
    extra = {
        "algo": args.algo,
        "timesteps": args.timesteps,
        "seed": seeds[0] if len(seeds) == 1 else None,
        "seeds": seeds,
        "seed_count": len(seeds),
        "successful_seed_count": len(model_paths),
        "failed_seeds": [run["seed"] for run in runs if run["status"] == "failed"],
        "eval_episodes_per_seed": args.eval_episodes,
        "evaluation_seed_policy": "common_frozen",
        "evaluation_seed_schedule": [
            EVAL_SEED_START + index for index in range(args.eval_episodes)
        ],
        "preregistered_main_schedule": (
            tuple(seeds) == PREREGISTERED_SEEDS
            and args.timesteps == 600_000
            and args.eval_episodes == PREREGISTERED_EVAL_EPISODES
        ),
        "model_path": str(model_paths[0]) if len(model_paths) == 1 else None,
        "model_paths": [str(path) for path in model_paths],
        "planned_model_paths": [run["model_path"] for run in runs],
        "runs": runs,
        "train_elapsed_sec": elapsed,
        "elapsed_sec": float(sum(run["elapsed_sec"] for run in runs)),
        "failure_stages": {
            str(run["seed"]): run["failure_stage"]
            for run in runs if run["status"] == "failed"
        },
        "aggregate": _aggregate(episodes),
        "host": _host_snapshot(),
        "repo_commit": _git_commit(Path(__file__).resolve().parents[2]),
    }
    _write_results(
        Path(args.out_dir),
        run_id,
        args.environment,
        [_task_spec(args)],
        episodes,
        extra=extra,
    )
    print(json.dumps(_main_payload(model_paths, seeds, elapsed, runs), indent=2))


if __name__ == "__main__":
    main()
