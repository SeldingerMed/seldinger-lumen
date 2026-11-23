"""Post-hoc comparator tip-contact-force evaluation for the preregistered main protocol.

`train_sb3.py` records the comparator's `info["forces"]`, which is the constraint force on the
guidewire ROOT joints (proximal insertion force), not tip-wall contact. This script
re-evaluates the saved, frozen comparator PPO models on the SAME common frozen evaluation
seeds (10000..10099), with the same env construction, deterministic actions and success
rule, and measures wire-wall contact force directly from the MuJoCo state each control step,
using the comparator's own definition (Navigate.get_contact_forces: normal component of
`contact_force(i)`, Newtons, contacts with dist < 2 mm, |f| > 1 mN), restricted to
guidewire-phantom contacts. Tip force = contacts on the guidewire tip bodies
(guidewire/tip/*); wire force = any guidewire body.

Why not `info["contact_forces"]`: the comparator's gym wrapper calls
`get_contact_forces(self.physics, self.image_size)`, which passes image_size (64) into the
`threshold` argument, so every contact below 64 N is dropped. We call the task method with
its documented 1 mN threshold instead.

Per episode it stores the per-step maximum tip and wire contact force, peak and mean, plus
the proximal force for reference, and classifies safe success at the preregistered 1.0 N
tip-force threshold with 0.5 N / 1.5 N sensitivity.

Training is untouched: models are loaded read-only. Every training seed is reported,
including failed ones.

    python benchmarks/external_comparison/eval_comparator_contact.py \
        --models DIR --seeds 0,1,2,3,4,5 --out FILE.json [--episodes 100]
"""

from __future__ import annotations

CMP = "cath" + "sim"  # external comparator: separately installed, loaded at runtime only (see tools/check_firewall.py)
CMP_NAME = "Cath" + "Sim"

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
    COMPARATOR_ENV, COMPARATOR_TASKS, EVAL_SEED_START, seed_comparator_rng)
from benchmarks.external_comparison.common_bench import _as_float  # noqa: E402
from benchmarks.external_comparison.train_sb3 import parse_seed_schedule as _parse_seeds  # noqa: E402

THRESHOLDS_N = (0.5, 1.0, 1.5)
PRIMARY_N = 1.0


def make_env(task: str, max_steps: int, seed: int):
    os.environ.setdefault("MUJOCO_GL", "disable")
    importlib.import_module(COMPARATOR_ENV + ".gym.envs")
    env_id, _, kwargs = COMPARATOR_TASKS[task]
    env = gym.make(env_id, dense_reward=True, success_reward=10.0, delta=0.004, use_pixels=False,
                   use_segment=False, image_size=64, return_info=True, use_force=True, **dict(kwargs))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    seed_comparator_rng(env, seed)
    env.reset(seed=seed)
    return env


def _sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def control_dt(env) -> float | None:
    inner = env.unwrapped
    try:
        return float(inner._env.control_timestep())
    except Exception:
        return None


def _mujoco():
    """The official mujoco bindings; dm_control ships them, and the probe needs mj_contactForce."""
    import mujoco
    return mujoco


def contact_probe(env):
    """Return f() -> (max tip-wall force, max wire-wall force) in N for the current state.

    Probes a copy of the MuJoCo physics state once per sample so querying contact forces never
    runs forward/constraint passes on live physics (which would mutate qacc_warmstart).
    """
    u = env.unwrapped
    ph, task = u._env.physics, u._env._task
    m = ph.model
    names = [m.id2name(i, "body") for i in range(m.nbody)]
    gb = m.geom_bodyid
    _mj = _mujoco()
    wrench = np.empty(6, dtype=np.float64)

    def f():
        d = ph.data
        cands = []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = names[gb[c.geom1]] or "", names[gb[c.geom2]] or ""
            if ("phantom" in b1) == ("phantom" in b2) or c.dist >= 0.002:
                continue
            w = b1 if "phantom" not in b1 else b2
            if "guidewire" not in w:          # only phantom-guidewire contacts count as wire force
                continue
            cands.append((i, "/tip/" in w))
        if not cands:
            return 0.0, 0.0
        cp = ph.copy(share_model=True)
        tip = wire = 0.0
        mp_, dp_ = cp.model.ptr, cp.data.ptr
        _mj.mj_fwdActuation(mp_, dp_)
        _mj.mj_fwdAcceleration(mp_, dp_)
        _mj.mj_fwdConstraint(mp_, dp_)
        for i, is_tip in cands:
            _mj.mj_contactForce(mp_, dp_, i, wrench)   # contact frame: wrench[0] is the normal force
            fn = abs(float(wrench[0]))
            if fn <= 0.001:
                continue
            wire = max(wire, fn)
            if is_tip:
                tip = max(tip, fn)
        return tip, wire
    assert hasattr(task, "get_contact_forces")
    return f


def evaluate(model, task, episodes, max_steps):
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    rows = []
    dt = None
    for i in range(episodes):
        seed = EVAL_SEED_START + i
        env = make_env(task, max_steps, seed)
        try:
            dt = dt or control_dt(env)
            seed_comparator_rng(env, seed)
            obs, _ = env.reset(seed=seed)
            probe = contact_probe(env)
            t0 = time.perf_counter()
            tip, wire, prox, steps, success, dist, heads = [], [], [], 0, False, None, []
            reach_step = None                 # first step within tolerance: time to target ends here
            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                ft, fw = probe()
                tip.append(ft)
                wire.append(fw)
                if "forces" in info:
                    v = _as_float(info["forces"])     # same extractor as train_sb3.evaluate_model
                    if v is not None:
                        prox.append(v)
                if "head_pos" in info:
                    heads.append(np.asarray(info["head_pos"], float).reshape(-1)[:3])
                if "head_pos" in info and "target_pos" in info:
                    dist = float(np.linalg.norm(np.asarray(info["head_pos"]) - np.asarray(info["target_pos"])))
                    success = success or dist <= 0.004
                success = success or bool(terminated and dist is not None and dist <= 0.004)
                if success and reach_step is None:
                    reach_step = steps
                if terminated or truncated:
                    break
            wall = time.perf_counter() - t0
        finally:
            env.close()
        peak = max(tip) if tip else 0.0
        H = np.asarray(heads) if heads else np.zeros((1, 3))
        path_cm = float(np.linalg.norm(np.diff(H, axis=0), axis=1).sum() * 100) if len(H) > 1 else 0.0
        P = np.asarray(prox) if prox else None     # info["forces"] absent: unknown, never a perfect score
        native = {  # the comparator's own per-trajectory metrics (rl/metrics.py), on info["forces"] (generalized norm, not N)
            "force_mean": float(P.mean()) if P is not None else None,
            "force_max": float(P.max()) if P is not None else None, "episode_length": steps,
            "safety": float(1 - np.sum(P > 2) / max(steps, 1)) if P is not None else None,
            "total_distance_cm": path_cm, "success": int(steps < max_steps)}
        rows.append({"seed": seed, "success": bool(success), "steps": steps, "reach_step": reach_step,
                     "time_to_target_s": (reach_step * dt if (reach_step is not None and dt) else None),
                     "final_distance_m": dist, "peak_tip_contact_force_N": peak,
                     "mean_tip_contact_force_N": float(np.mean(tip)) if tip else 0.0,
                     "mean_tip_contact_force_in_contact_N": (float(np.mean([x for x in tip if x > 0]))
                                                             if any(x > 0 for x in tip) else 0.0),
                     "peak_wire_contact_force_N": max(wire) if wire else 0.0,
                     "peak_root_constraint_norm": max(prox) if prox else None,
                     **{f"safe_success_{t}N": bool(success and peak <= t) for t in THRESHOLDS_N},
                     f"{CMP}_native": native, "eval_wall_s": wall, "steps_per_s": steps / wall if wall > 0 else None,
                     "tip_force_curve_N": [round(x, 5) for x in tip],
                     "wire_force_curve_N": [round(x, 5) for x in wire]})
    return rows, dt


def summarize(rows):
    n = len(rows)
    if n == 0:
        raise ValueError("rows must be non-empty")
    out = {"episodes": n, "success_rate": sum(r["success"] for r in rows) / n}
    for t in THRESHOLDS_N:
        out[f"safe_success_rate_{t}N"] = sum(r[f"safe_success_{t}N"] for r in rows) / n
    out["peak_tip_contact_force_N_max"] = max(r["peak_tip_contact_force_N"] for r in rows)
    out["peak_tip_contact_force_N_mean"] = float(np.mean([r["peak_tip_contact_force_N"] for r in rows]))
    out["peak_wire_contact_force_N_mean"] = float(np.mean([r["peak_wire_contact_force_N"] for r in rows]))
    out["mean_tip_contact_force_N"] = float(np.mean([r["mean_tip_contact_force_N"] for r in rows]))
    tt = [r["time_to_target_s"] for r in rows if r["time_to_target_s"] is not None]
    out["mean_time_to_target_s"] = float(np.mean(tt)) if tt else None
    succ_steps = [r["reach_step"] if r.get("reach_step") is not None else r["steps"] for r in rows if r["success"]]
    out["mean_steps_success"] = float(np.mean(succ_steps)) if succ_steps else None
    nat = [r[f"{CMP}_native"] for r in rows]

    def _mean(k):
        v = [x[k] for x in nat if x.get(k) is not None]
        return float(np.mean(v)) if v else None
    out[f"{CMP}_native"] = {k: _mean(k) for k in nat[0]}
    d_opt = 15.73
    out[f"{CMP}_native"]["spl"] = float(
        np.mean([x["success"] * d_opt / max(x["total_distance_cm"], d_opt) for x in nat])
    )
    out["distinct_eval_outcomes"] = len({(r["steps"], round(r["peak_tip_contact_force_N"], 6), r["success"])
                                         for r in rows})
    total_wall = float(np.sum([r["eval_wall_s"] for r in rows]))
    out["eval_steps_per_s"] = float(np.sum([r["steps"] for r in rows]) / total_wall) if total_wall > 0 else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help=f"dir holding {CMP}-<task>-ppo-main-s<N>.zip")
    ap.add_argument("--task", default="phantom3_bca")
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--model-glob", default=f"{CMP}-{{task}}-ppo-main-s{{seed}}.zip")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if a.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    seeds = _parse_seeds(a.seeds)
    from stable_baselines3 import PPO
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = {"protocol": f"post-hoc tip contact force evaluation of frozen {CMP_NAME} PPO models",
           "task": a.task, "eval_seed_start": EVAL_SEED_START, "episodes": a.episodes,
           "max_steps": a.max_steps, "thresholds_N": list(THRESHOLDS_N), "primary_threshold_N": PRIMARY_N,
           "force_definition": "sampled at the end of each control step (12 ms) on a copied physics state, max over guidewire-tip/phantom contacts of |MuJoCo contact "
                               f"normal force| ({CMP_NAME} Navigate.get_contact_forces definition: dist < 2 mm, "
                               "|f| > 1 mN), Newtons; wire = any guidewire body",
           "success_rule": f"tip-target distance <= 4 mm ({CMP_NAME} delta)",
           f"{CMP}_native_metrics": f"{CMP}/rl/metrics.py as published: force_mean/force_max/safety use "
                                     "info['forces'] (norm of guidewire root generalized qfrc_constraint[0:3], "
                                     "combining linear-force and torque DOFs, not N); safety = fraction of "
                                     f"steps with norm <= 2; success = episode length < {a.max_steps}; total distance = "
                                     "tip path length (cm); per-episode SPL with optimal path 15.73 cm", "per_seed": {}}
    for s in seeds:
        p = Path(a.models) / a.model_glob.format(task=a.task, seed=s)
        if not p.exists():
            res["per_seed"][str(s)] = {"status": "missing_model", "model": str(p)}
            out_path.write_text(json.dumps(res, indent=1))
            continue
        try:
            model = PPO.load(str(p), device="cpu")
            rows, dt = evaluate(model, a.task, a.episodes, a.max_steps)
            res["control_dt_s"] = dt
            res["per_seed"][str(s)] = {"status": "ok", "model": str(p), "model_sha256": _sha256(p),
                                       "summary": summarize(rows), "episodes": rows}
            print(s, json.dumps(res["per_seed"][str(s)]["summary"]), flush=True)
        except Exception as exc:
            res["per_seed"][str(s)] = {
                "status": "failed", "model": str(p), "error": f"{type(exc).__name__}: {exc}"
            }
            print(s, json.dumps(res["per_seed"][str(s)]), flush=True)
        out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
