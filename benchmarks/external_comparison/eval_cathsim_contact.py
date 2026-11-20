"""Post-hoc CathSim tip-contact-force evaluation for the preregistered main protocol.

`train_sb3.py` records CathSim's `info["forces"]`, which is the constraint force on the
guidewire ROOT joints (proximal insertion force), not tip-wall contact. This script
re-evaluates the saved, frozen CathSim PPO models on the SAME common frozen evaluation
seeds (10000..10099), with the same env construction, deterministic actions and success
rule, and additionally enables `use_contact_forces=True` to record CathSim's own per-contact
normal forces (MuJoCo `contact_force(i)[0][0]`, SI units, Newtons; contacts within 2 mm,
|f| > 1 mN, CathSim's defaults). Per episode it stores the per-step maximum contact force,
peak and mean, plus the proximal force for reference, and classifies safe success at the
preregistered 1.0 N threshold with 0.5 N / 1.5 N sensitivity.

Training is untouched: models are loaded read-only. Every training seed is reported,
including failed ones.

    python benchmarks/external_comparison/eval_cathsim_contact.py \
        --models DIR --seeds 0,1,2,3,4,5 --out FILE.json [--episodes 100]
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.external_comparison.train_sb3 import (  # noqa: E402
    COMPARATOR_ENV, COMPARATOR_TASKS, EVAL_SEED_START)

THRESHOLDS_N = (0.5, 1.0, 1.5)
PRIMARY_N = 1.0


def make_env(task: str, max_steps: int, seed: int):
    os.environ.setdefault("MUJOCO_GL", "disable")
    importlib.import_module(COMPARATOR_ENV + ".gym.envs")
    env_id, _, kwargs = COMPARATOR_TASKS[task]
    env = gym.make(env_id, dense_reward=True, success_reward=10.0, delta=0.004, use_pixels=False,
                   use_segment=False, image_size=64, return_info=True, use_force=True,
                   use_contact_forces=True, **dict(kwargs))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return env


def control_dt(env) -> float | None:
    inner = env.unwrapped
    try:
        return float(inner._env.control_timestep())
    except Exception:
        return None


def evaluate(model, task, episodes, max_steps):
    rows = []
    dt = None
    for i in range(episodes):
        seed = EVAL_SEED_START + i
        env = make_env(task, max_steps, seed)
        dt = dt or control_dt(env)
        obs, _ = env.reset(seed=seed)
        t0 = time.perf_counter()
        tip, prox, steps, success, dist, heads = [], [], 0, False, None, []
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            cf = info.get("contact_forces") or {}
            f = [abs(float(x)) for x in cf.get("force", [])]
            tip.append(max(f) if f else 0.0)
            if "forces" in info:
                prox.append(float(np.asarray(info["forces"]).reshape(-1)[0]))
            if "head_pos" in info:
                heads.append(np.asarray(info["head_pos"], float).reshape(-1)[:3])
            if "head_pos" in info and "target_pos" in info:
                dist = float(np.linalg.norm(np.asarray(info["head_pos"]) - np.asarray(info["target_pos"])))
                success = success or dist <= 0.004
            success = success or bool(terminated and dist is not None and dist <= 0.004)
            if terminated or truncated:
                break
        wall = time.perf_counter() - t0
        env.close()
        peak = max(tip) if tip else 0.0
        H = np.asarray(heads) if heads else np.zeros((1, 3))
        path_cm = float(np.linalg.norm(np.diff(H, axis=0), axis=1).sum() * 100) if len(H) > 1 else 0.0
        P = np.asarray(prox) if prox else np.zeros(1)
        native = {  # CathSim's own per-trajectory metrics (cathsim/rl/metrics.py), on info["forces"]
            "force_mean": float(P.mean()), "force_max": float(P.max()), "episode_length": steps,
            "safety": float(1 - np.sum(P > 2) / max(steps, 1)), "total_distance_cm": path_cm,
            "success": int(steps < 300)}
        rows.append({"seed": seed, "success": bool(success), "steps": steps,
                     "time_to_target_s": (steps * dt if (success and dt) else None),
                     "final_distance_m": dist, "peak_tip_contact_force_N": peak,
                     "mean_tip_contact_force_N": float(np.mean(tip)) if tip else 0.0,
                     "mean_tip_contact_force_in_contact_N": (float(np.mean([x for x in tip if x > 0]))
                                                             if any(x > 0 for x in tip) else 0.0),
                     "peak_proximal_force_N": max(prox) if prox else None,
                     **{f"safe_success_{t}N": bool(success and peak <= t) for t in THRESHOLDS_N},
                     "cathsim_native": native, "eval_wall_s": wall, "steps_per_s": steps / wall if wall > 0 else None,
                     "tip_force_curve_N": [round(x, 5) for x in tip]})
    return rows, dt


def summarize(rows):
    n = len(rows)
    out = {"episodes": n, "success_rate": sum(r["success"] for r in rows) / n}
    for t in THRESHOLDS_N:
        out[f"safe_success_rate_{t}N"] = sum(r[f"safe_success_{t}N"] for r in rows) / n
    out["peak_tip_contact_force_N_max"] = max(r["peak_tip_contact_force_N"] for r in rows)
    out["peak_tip_contact_force_N_mean"] = float(np.mean([r["peak_tip_contact_force_N"] for r in rows]))
    out["mean_tip_contact_force_N"] = float(np.mean([r["mean_tip_contact_force_N"] for r in rows]))
    tt = [r["time_to_target_s"] for r in rows if r["time_to_target_s"] is not None]
    out["mean_time_to_target_s"] = float(np.mean(tt)) if tt else None
    out["mean_steps_success"] = float(np.mean([r["steps"] for r in rows if r["success"]])) if tt else None
    nat = [r["cathsim_native"] for r in rows]
    out["cathsim_native"] = {k: float(np.mean([x[k] for x in nat])) for k in nat[0]}
    agg = sum(x["total_distance_cm"] for x in nat) / n
    out["cathsim_native"]["spl"] = float(sum(x["success"] for x in nat) / n * 15.73 / max(agg, 15.73))
    out["distinct_eval_outcomes"] = len({(r["steps"], round(r["peak_tip_contact_force_N"], 6), r["success"])
                                         for r in rows})
    out["eval_steps_per_s"] = float(np.sum([r["steps"] for r in rows]) / np.sum([r["eval_wall_s"] for r in rows]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="dir holding cathsim-<task>-ppo-main-s<N>.zip")
    ap.add_argument("--task", default="phantom3_bca")
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--model-glob", default="cathsim-{task}-ppo-main-s{seed}.zip")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from stable_baselines3 import PPO
    res = {"protocol": "post-hoc tip contact force evaluation of frozen CathSim PPO models",
           "task": a.task, "eval_seed_start": EVAL_SEED_START, "episodes": a.episodes,
           "max_steps": a.max_steps, "thresholds_N": list(THRESHOLDS_N), "primary_threshold_N": PRIMARY_N,
           "force_definition": "max over active contacts of |MuJoCo per-contact normal force| "
                               "(CathSim Navigate.get_contact_forces: dist < 2 mm, |f| > 1 mN), Newtons",
           "success_rule": "tip-target distance <= 4 mm (CathSim delta)",
           "cathsim_native_metrics": "cathsim/rl/metrics.py as published: force_mean/force_max/safety use "
                                     "info['forces'] (guidewire root constraint force, N); safety = fraction of "
                                     "steps with force <= 2 N; success = episode length < 300; total distance = "
                                     "tip path length (cm); SPL with optimal path 15.73 cm", "per_seed": {}}
    for s in [int(x) for x in a.seeds.split(",") if x]:
        p = Path(a.models) / a.model_glob.format(task=a.task, seed=s)
        if not p.exists():
            res["per_seed"][str(s)] = {"status": "missing_model", "model": str(p)}
            continue
        model = PPO.load(str(p), device="cpu")
        rows, dt = evaluate(model, a.task, a.episodes, a.max_steps)
        res["control_dt_s"] = dt
        res["per_seed"][str(s)] = {"status": "ok", "model": str(p), "summary": summarize(rows), "episodes": rows}
        print(s, json.dumps(res["per_seed"][str(s)]["summary"]), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
