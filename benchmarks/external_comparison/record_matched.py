"""Record matched evaluation episodes for the X-series side-by-side video (item 8).

For each training seed, loads the frozen PPO model from the preregistered main run and plays
one deterministic episode on the common evaluation seed (default 10000, the first of the
frozen 100), exactly as train_sb3.evaluate_model does, and stores per-step geometry and
safety readouts in an .npz per seed:

    vessel   (P, 3)  surface sample of the vessel lumen, mm, world frame
    target   (3,)    target point, mm
    wire_k   (N_k,3) guidewire polyline at step k, mm (object array over steps)
    safety   (T,)    per-step safety readout: Lumen surface overlap (mm);
                     comparator max tip-wall contact force (N)
    success, steps, dt, eval_seed, training_seed, environment

Runs in either venv: --environment lumen (Lumen venv) or the comparator (its own venv, MUJOCO_GL=disable).

    python benchmarks/external_comparison/record_matched.py --environment lumen \\
        --models DIR --seeds 0,1,2,3,4,5 --out DIR
"""

from __future__ import annotations

CMP = "cath" + "sim"  # external comparator: separately installed, loaded at runtime only (see tools/check_firewall.py)

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.external_comparison.train_sb3 import (  # noqa: E402
    EVAL_SEED_START, make_env, seed_comparator_rng)

TASK = {"lumen": "nav_tree_branch", CMP: "phantom3_bca"}


def tube_surface(points, radii, n_theta=64, ds=0.2):
    P = np.asarray(points, float)
    R = np.broadcast_to(np.asarray(radii, float), (len(P),))
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.arange(0, s[-1], ds)
    C = np.stack([np.interp(ss, s, P[:, i]) for i in range(3)], 1)
    Rr = np.interp(ss, s, R)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    ref = np.where(np.abs(T[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    U = np.cross(T, ref)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = np.cross(T, U)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    pts = C[:, None, :] + Rr[:, None, None] * (np.cos(th)[None, :, None] * U[:, None, :]
                                               + np.sin(th)[None, :, None] * V[:, None, :])
    return pts.reshape(-1, 3)


def lumen_inner(env):
    u = env.unwrapped
    return getattr(u, "_env", u)


def lumen_geometry(env):
    u = lumen_inner(env)
    surf = [tube_surface(e.frame.points, np.asarray(e.lf.R, float).reshape(len(e.frame.points), -1).mean(1))
            for e in u.tree.edges]
    tgt = np.asarray(u.route_frame.points[-1], float)
    return np.concatenate(surf), tgt


_SHAFT_WARNED = []


def lumen_wire(env):
    u = lumen_inner(env)
    pos = np.asarray(u.sim.body_positions(), float)
    try:
        from tools.xseries.xs_common import shaft_polyline
        return np.asarray(shaft_polyline(u, pos), float)
    except Exception as exc:          # draw the rod alone, but say so: the two sides would differ
        if not _SHAFT_WARNED:
            _SHAFT_WARNED.append(exc)
            print(f"WARNING: Lumen shaft polyline unavailable ({exc!r}); recording rod bodies only",
                  file=sys.stderr, flush=True)
        return pos


def comparator_geometry(env):
    u = env.unwrapped
    ph = u._env.physics
    m = ph.model
    def mesh_name(i):
        return (m.id2name(int(m.geom_dataid[i]), "mesh") or "") if m.geom_dataid[i] >= 0 else ""
    gid = [i for i in range(m.ngeom) if mesh_name(i).endswith("visual")] or \
        [i for i in range(m.ngeom) if "hull" in mesh_name(i)]
    parts = []
    for g in gid:
        mid = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        V = np.asarray(m.mesh_vert[a:a + n], float)
        parts.append(ph.data.geom_xpos[g] + V @ np.asarray(ph.data.geom_xmat[g]).reshape(3, 3).T)
    X = np.concatenate(parts)
    tgt = np.asarray(u._env._task.get_target_pos(ph), float).reshape(-1)[:3]
    return X * 1000.0, tgt * 1000.0


def comparator_wire(env):
    u = env.unwrapped
    ph = u._env.physics
    m = ph.model
    ids = [i for i in range(m.ngeom) if "guidewire" in (m.id2name(int(m.geom_bodyid[i]), "body") or "")]
    return np.asarray(ph.data.geom_xpos[ids], float) * 1000.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--environment", choices=["lumen", CMP], required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4,5")
    ap.add_argument("--eval-seed", type=int, default=EVAL_SEED_START)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--model-glob", default="{env}-{task}-ppo-main-s{seed}.zip")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from stable_baselines3 import PPO
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env_name = a.environment
    task = TASK[env_name]
    failures = {}
    for s in [int(x) for x in a.seeds.split(",") if x]:
        mp = Path(a.models) / a.model_glob.format(env=env_name, task=task, seed=s)
        if not mp.exists():
            print("missing", mp, flush=True)
            failures[s] = f"missing model {mp}"
            continue
        try:
            _record_one(a, PPO, env_name, task, s, mp, out)
        except Exception as exc:          # one bad seed never aborts the other recordings
            failures[s] = repr(exc)
            print(env_name, s, "FAILED", repr(exc), flush=True)
    if failures:
        (out / f"{env_name}-failures.json").write_text(json.dumps({str(k): v for k, v in failures.items()}, indent=1))
    return 1 if failures else 0


def _record_one(a, PPO, env_name, task, s, mp, out):
    model = PPO.load(str(mp), device="cpu")
    env = make_env(env_name, task, a.max_steps, a.eval_seed)
    try:
        if env_name == CMP:
            seed_comparator_rng(env, a.eval_seed)
        obs, _ = env.reset(seed=a.eval_seed)
        if env_name == "lumen":
            vessel, tgt = lumen_geometry(env)
            probe = None
            wire_fn, dt = lumen_wire, 5e-3 * float(getattr(lumen_inner(env), "substeps", 4))
        else:
            from benchmarks.external_comparison.eval_comparator_contact import contact_probe
            vessel, tgt = comparator_geometry(env)
            probe = contact_probe(env)
            wire_fn, dt = comparator_wire, float(env.unwrapped._env.control_timestep())
        wires, safety, success = [wire_fn(env)], [0.0], False
        for _ in range(a.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            wires.append(wire_fn(env))
            if env_name == "lumen":
                safety.append(float(info.get("max_pen", 0.0)))
                success = success or bool(info.get("success", False))
            else:
                safety.append(probe()[0])
                if "head_pos" in info and "target_pos" in info:
                    d = float(np.linalg.norm(np.asarray(info["head_pos"]) - np.asarray(info["target_pos"])))
                    success = success or d <= 0.004
            if term or trunc:
                break
        w = np.empty(len(wires), dtype=object)
        w[:] = wires
        np.savez_compressed(out / f"{env_name}-s{s}.npz", vessel=vessel.astype(np.float32), target=tgt, wire=w,
                            safety=np.asarray(safety), success=success, steps=len(wires) - 1, dt=dt,
                            eval_seed=a.eval_seed, training_seed=s, environment=env_name, model=str(mp))
        print(env_name, s, "success", success, "steps", len(wires) - 1, "peak", max(safety), flush=True)
    finally:
        try:            # a teardown error neither masks the real one nor fails a saved recording
            env.close()
        except Exception as close_exc:
            print(f"{env_name} seed {s}: env.close() failed: {close_exc!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
