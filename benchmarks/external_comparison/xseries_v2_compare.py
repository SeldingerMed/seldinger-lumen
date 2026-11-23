"""X series v2 summary: wall-clock time to 90% success on the frozen evaluation seeds (XSERIES_V2_PROTOCOL.md).

Inputs per (simulator, configuration, task, training seed):
- the training result JSON written by train_sb3.py (training throughput, env-steps/s over the timed,
  uninterrupted segment; see xseries_compare.train_rate), and
- the learning curve written by checkpoint_curve.py (first checkpoint with success >= 90%).

Wall-clock to target = steps to target / that run's training throughput. A run that never reaches the
threshold is reported as not reached with its best observed success; it is not imputed, and it is
excluded from the mean (the count of runs that reached is reported beside every mean).

    python -m benchmarks.external_comparison.xseries_v2_compare --spec runs.json --out results/xseries-v2
runs.json: [{"sim": "lumen", "config": "single", "task": "bca", "seed": 0,
             "train": ".../lumen-phantom3_bca-ppo-main-s0.json", "curve": ".../curves/....json"}, ...]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from benchmarks.external_comparison.xseries_compare import ci, train_rate

CMP = "cath" + "sim"
CMP_NAME = "Cath" + "Sim"
SIM_NAME = {"lumen": "Lumen", CMP: CMP_NAME}


def run_row(spec: dict) -> dict:
    t = json.loads(Path(spec["train"]).read_text())
    c = json.loads(Path(spec["curve"]).read_text())
    ident = c["identity"]
    if ident["training_seed"] != spec["seed"] or not ident["task"].endswith(spec["task"]):
        raise ValueError(f"curve {spec['curve']} is for seed {ident['training_seed']} / {ident['task']}, "
                         f"not seed {spec['seed']} / {spec['task']}")
    sps, _ = train_rate(t)
    pts = c["points"]
    steps = c.get("first_reach_steps")
    best = max(pts, key=lambda p: (p["exact"], p["success_rate"])) if pts else None
    eval_steps = sum(p["eval_env_steps"] for p in pts)
    eval_wall = sum(p["eval_wall_s"] for p in pts)
    runs = t["extra"].get("runs") or [{}]
    return {
        **{k: spec[k] for k in ("sim", "config", "task", "seed")},
        "n_envs": int(t["extra"].get("n_envs") or spec.get("n_envs") or 1),
        "reached": steps is not None,
        "steps_to_target": steps,
        "train_env_steps_per_s": sps,
        "hours_to_target": (steps / sps / 3600) if (steps is not None and sps) else None,
        "train_hours_projected": bool(runs[0].get("resumed_from_step")),
        "best_success": best["success_rate"] if best else None,
        "best_success_exact": bool(best and best["exact"]),
        "eval_env_steps_per_s": (eval_steps / eval_wall) if eval_wall > 0 else None,
        "checkpoints_scored": len(pts),
        "curve": [{"steps": p["steps"], "success": p["success_rate"], "exact": p["exact"]} for p in pts],
    }


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["sim"], r["config"], r["task"])].append(r)
    out = {}
    for (sim, config, task), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["seed"])
        out[f"{sim}/{config}/{task}"] = {
            "sim": sim, "config": config, "task": task, "seeds": [r["seed"] for r in rs],
            "reached": sum(r["reached"] for r in rs), "runs": len(rs),
            "hours_to_target": ci([r["hours_to_target"] for r in rs], nonneg=True),
            "steps_to_target": ci([r["steps_to_target"] for r in rs], nonneg=True),
            "train_env_steps_per_s": ci([r["train_env_steps_per_s"] for r in rs], nonneg=True),
            "eval_env_steps_per_s": ci([r["eval_env_steps_per_s"] for r in rs], nonneg=True),
            "n_envs": sorted({r["n_envs"] for r in rs}),
        }
    return out


def _fmt(c: dict, nd: int = 2) -> str:
    if c["mean"] is None:
        return "not reached"
    s = f"{c['mean']:.{nd}f}"
    if c["lo"] is not None:
        s += f" [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"
    return s


def markdown(summary: dict, rows: list[dict]) -> str:
    md = ["| Simulator | Configuration | Target | Reached 90% | Hours to 90% (mean, 95% CI) | "
          "Steps to 90% | Train env-steps/s | Eval env-steps/s |", "|---|---|---|---|---|---|---|---|"]
    for g in summary.values():
        cfg = g["config"] + (f" (E={','.join(map(str, g['n_envs']))})" if g["n_envs"] != [1] else "")
        md.append(f"| {SIM_NAME[g['sim']]} | {cfg} | {g['task'].upper()} | {g['reached']}/{g['runs']} | "
                  f"{_fmt(g['hours_to_target'])} | {_fmt(g['steps_to_target'], 0)} | "
                  f"{_fmt(g['train_env_steps_per_s'], 1)} | {_fmt(g['eval_env_steps_per_s'], 1)} |")
    md += ["", "Per training seed:", "", "| Simulator | Configuration | Target | Seed | Steps to 90% | Hours to 90% | "
           "Best success |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["sim"], r["config"], r["task"], r["seed"])):
        hrs = "not reached" if r["hours_to_target"] is None else f"{r['hours_to_target']:.2f}"
        stp = "not reached" if r["steps_to_target"] is None else f"{r['steps_to_target']:,}"
        best = "n/a" if r["best_success"] is None else f"{100 * r['best_success']:.0f}%"
        md.append(f"| {SIM_NAME[r['sim']]} | {r['config']} | {r['task'].upper()} | {r['seed']} | {stp} | {hrs} | {best} |")
    md += ["", "Hours to 90% = steps to the first checkpoint at >= 90% success on evaluation seeds 10000-10099 "
           "divided by the run's measured training throughput. Means are over the runs that reached 90%."]
    return "\n".join(md) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="X series v2: time to 90% success")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True, help="output prefix (.json and .md are written)")
    a = ap.parse_args(argv)
    specs = json.loads(Path(a.spec).read_text())
    rows = [run_row(s) for s in specs]
    summary = summarize(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps({"summary": summary, "runs": rows}, indent=1))
    Path(a.out + ".md").write_text(markdown(summary, rows))
    print(markdown(summary, rows))


if __name__ == "__main__":
    main()
