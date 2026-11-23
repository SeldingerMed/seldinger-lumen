"""Learning curve from saved PPO snapshots: success on the frozen evaluation seeds at every checkpoint.

X series v2 headline (XSERIES_V2_PROTOCOL.md): steps to target = steps at the first checkpoint whose
success on the 100 frozen evaluation seeds (10000..10099, deterministic actions) is >= 90%.

Each checkpoint is evaluated in schedule order. Once more than ``episodes - ceil(threshold * episodes)``
episodes have failed, the checkpoint can no longer reach the threshold and its evaluation stops
(``exact: false``); the decision "below threshold" is exact and the success over the evaluated prefix
is kept for plotting. A checkpoint that reaches the threshold is always evaluated on every seed.

The snapshot at the full budget is not used (SB3's CheckpointCallback fires before the last rollout is
consumed by an update); the final saved model stands for the full budget instead.

Results are written after every checkpoint, and a rerun skips checkpoints already in the output file,
so an interrupted pass resumes where it stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

CKPT = re.compile(r"ckpt_(\d+)_steps\.zip$")


def snapshots(ckpt_dir: Path, budget: int, final_model: Path | None) -> list[tuple[int, Path]]:
    """(steps, path) in step order: snapshots strictly before the budget, then the final model."""
    pts = []
    for p in ckpt_dir.iterdir():
        m = CKPT.search(p.name)
        if m and not p.name.startswith("._") and int(m.group(1)) < budget:
            pts.append((int(m.group(1)), p))
    pts.sort()
    if final_model is not None:
        pts.append((budget, final_model))
    return pts


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_reach(points: list[dict], threshold: float) -> int | None:
    for p in sorted(points, key=lambda q: q["steps"]):
        if p["exact"] and p["success_rate"] >= threshold:
            return p["steps"]
    return None


def _write(out: Path, rec: dict) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, out)


def run(a, evaluate=None, load=None) -> dict:
    from benchmarks.external_comparison import train_sb3 as T

    evaluate = evaluate or T.evaluate_model
    load = load or (lambda p: T.ALGOS["ppo"].load(str(p), device="cpu"))
    out = Path(a.out)
    rec = json.loads(out.read_text()) if out.exists() else None
    ident = {"environment": a.environment, "task": a.task, "training_seed": a.seed, "episodes": a.episodes,
             "max_steps": a.max_steps, "threshold": a.threshold, "eval_seed_start": T.EVAL_SEED_START,
             "budget": a.timesteps}
    if rec is None or rec.get("identity") != ident:
        if rec is not None:
            raise ValueError(f"{out} was written for a different evaluation: {rec.get('identity')}")
        rec = {"schema": "xseries.checkpoint_curve/v1", "identity": ident, "points": []}
    done = {p["steps"] for p in rec["points"]}
    max_fail = a.episodes - math.ceil(a.threshold * a.episodes)
    eargs = argparse.Namespace(algo="ppo", environment=a.environment, eval_episodes=a.episodes,
                               max_steps=a.max_steps, seed=a.seed, task=a.task, timesteps=a.timesteps)
    final = Path(a.final_model) if a.final_model else None
    for steps, path in snapshots(Path(a.ckpt_dir), a.timesteps, final):
        if steps in done:
            continue
        model = load(path)
        t0 = time.perf_counter()
        eps = evaluate(eargs, model, training_seed=a.seed, model_id=path.name, max_failures=max_fail)
        wall = time.perf_counter() - t0
        n, succ = len(eps), sum(bool(e.success) for e in eps)
        rec["points"].append({
            "steps": steps, "model": path.name, "model_sha256": sha256(path),
            "episodes_evaluated": n, "successes": succ, "success_rate": succ / n if n else 0.0,
            "exact": n == a.episodes, "crashed": sum(bool(getattr(e, "crashed", False)) for e in eps),
            "eval_env_steps": int(sum(e.steps for e in eps)), "eval_wall_s": wall,
        })
        rec["points"].sort(key=lambda p: p["steps"])
        rec["first_reach_steps"] = first_reach(rec["points"], a.threshold)
        _write(out, rec)
        print(f"{a.task} s{a.seed} {steps:>7} {succ}/{n}{'' if n == a.episodes else ' (stopped)'} "
              f"{wall:.0f}s", flush=True)
        if a.stop_at_first and rec["first_reach_steps"] is not None:
            break
    rec["first_reach_steps"] = first_reach(rec["points"], a.threshold)
    _write(out, rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Learning curve from saved PPO snapshots.")
    ap.add_argument("--environment", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True, help="training seed of the run")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--final-model", default="", help="model saved at the end of training (full budget)")
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--stop-at-first", action="store_true", help="stop after the first checkpoint at threshold")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.episodes <= 0 or a.max_steps <= 0 or a.timesteps <= 0:
        raise ValueError("--episodes, --max-steps and --timesteps must be positive")
    if not 0.0 < a.threshold <= 1.0:
        raise ValueError("--threshold must be in (0, 1]")
    run(a)


if __name__ == "__main__":
    main()
