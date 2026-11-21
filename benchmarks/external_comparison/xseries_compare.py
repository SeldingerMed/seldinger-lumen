"""Lumen vs CathSim head-to-head summary for the preregistered main protocol (X series).

Reads the per-training-seed result JSONs written by train_sb3.py (both environments) and
the post-hoc CathSim tip-contact-force evaluation (eval_cathsim_contact.py), and writes one
JSON + Markdown table. Statistical unit: the training seed (the --seeds set, same seeds in both environments); each
seed's policy is scored on the same 100 frozen evaluation seeds (10000..10099). Every
training seed is listed, including failures. Means carry two-sided 95% t-intervals across
training seeds.

Safety is reported on each simulator's own physical endpoint, at its stated threshold:
  * Lumen: device-wall surface overlap <= 0.3 mm (the contact barrier width d_hat).
  * CathSim: peak tip contact force <= 1.0 N (sensitivity 0.5 N and 1.5 N), from CathSim's
    own per-contact MuJoCo normal forces in Newtons.
The two thresholds are not cross-calibrated; the table does not equate them.

    python benchmarks/external_comparison/xseries_compare.py --bench DIR --contact FILE --out PREFIX
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
LUMEN_DT = 4 * 5e-3          # TreeNavEnv: substeps=4 x 5 ms


def ci(vals, bounded=False):
    """Mean with a two-sided 95% t-interval; rates are clipped to [0, 1]."""
    v = [x for x in vals if x is not None]
    n = len(v)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = sum(v) / n
    if n == 1:
        return {"mean": m, "lo": None, "hi": None, "n": 1}
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    h = T975.get(n - 1, 1.96) * sd / math.sqrt(n)
    lo, hi = m - h, m + h
    if bounded:
        lo, hi = max(0.0, lo), min(1.0, hi)
    return {"mean": m, "lo": lo, "hi": hi, "n": n}


def lumen_seed(path):
    j = json.load(open(path))
    a = j["aggregate"][0]
    eps = j["episodes"]
    succ = [e for e in eps if e["success"]]
    return {"seed": j["extra"]["seed"], "status": "ok",
            "success_rate": a["success_rate"], "safe_success_rate": a["native_safety_pass_rate"],
            "peak_overlap_mm": max(e["safety_value"] for e in eps),
            "peak_wall_load_sim_units": a["max_wall_load"],
            "time_to_target_s": (sum(e["steps"] for e in succ) / len(succ) * LUMEN_DT) if succ else None,
            "train_hours": j["extra"]["train_elapsed_sec"] / 3600,
            "train_env_steps_per_s": j["extra"]["timesteps"] / j["extra"]["train_elapsed_sec"],
            "eval_env_steps_per_s": a["steps_per_second"],
            "distinct_eval_outcomes": len({(e["steps"], round(e["safety_value"], 6), e["success"]) for e in eps}),
            "eval_episodes": len(eps)}


def cathsim_seed(train_path, contact):
    j = json.load(open(train_path))
    s = int(j["extra"]["seed"])
    a = j["aggregate"][0]
    row = {"seed": s, "status": "ok", "train_hours": j["extra"]["train_elapsed_sec"] / 3600,
           "train_env_steps_per_s": j["extra"]["timesteps"] / j["extra"]["train_elapsed_sec"],
           "resumed_from_step": [r.get("resumed_from_step") for r in j["extra"].get("runs", [])],
           "success_rate_train_eval": a["success_rate"], "eval_env_steps_per_s": a["steps_per_second"]}
    c = (contact or {}).get("per_seed", {}).get(str(s))
    if c and c.get("status") == "ok":
        m = c["summary"]
        dt = contact.get("control_dt_s")
        row.update({"success_rate": m["success_rate"],
                    "safe_success_rate": m["safe_success_rate_1.0N"],
                    "safe_success_rate_0.5N": m["safe_success_rate_0.5N"],
                    "safe_success_rate_1.5N": m["safe_success_rate_1.5N"],
                    "peak_tip_force_N": m["peak_tip_contact_force_N_max"],
                    "mean_peak_tip_force_N": m["peak_tip_contact_force_N_mean"],
                    "mean_tip_force_N": m["mean_tip_contact_force_N"],
                    "time_to_target_s": m["mean_time_to_target_s"],
                    "control_dt_s": dt, "distinct_eval_outcomes": m["distinct_eval_outcomes"],
                    "cathsim_native": m["cathsim_native"], "eval_episodes": m["episodes"]})
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--contact", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4,5", help="training seeds compared (same set in both sims)")
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",")]
    B = Path(a.bench)
    contact = json.load(open(a.contact)) if a.contact and Path(a.contact).exists() else None
    L = [lumen_seed(p) for p in sorted(glob.glob(str(B / "lumen-nav_tree_branch-ppo-main-s*.json")))]
    C = [cathsim_seed(p, contact) for p in sorted(glob.glob(str(B / "cathsim-phantom3_bca-ppo-main-s*.json")))]
    L_all = list(L)
    L = [r for r in L if r["seed"] in seeds]
    C = [r for r in C if r["seed"] in seeds]
    have = {r["seed"] for r in C}
    C += [{"seed": s, "status": "not_finished"} for s in seeds if s not in have]
    C.sort(key=lambda r: r["seed"])
    keys = ["success_rate", "safe_success_rate", "time_to_target_s", "train_hours", "train_env_steps_per_s",
            "eval_env_steps_per_s"]
    res = {"protocol": f"lumen-common-endovascular-bench main_budget: PPO, 600k env steps, training seeds "
                       f"{', '.join(map(str, seeds))} in both simulators, 100 common frozen eval seeds from 10000, "
                       "deterministic policy",
           "training_seeds": seeds,
           "task_class": "branch_or_arch_navigation",
           "tasks": {"lumen": "Lumen/NavTreeBranch-v0", "cathsim": "CathSim phantom3 / bca"},
           "thresholds": {"lumen": "surface overlap <= 0.3 mm", "cathsim": "peak tip contact force <= 1.0 N "
                          "(sensitivity 0.5 N, 1.5 N)"},
           "lumen": {"per_seed": L, "summary": {k: ci([r.get(k) for r in L], "rate" in k) for k in keys}},
           "cathsim": {"per_seed": C, "summary": {k: ci([r.get(k) for r in C if r["status"] == "ok"], "rate" in k)
                                                   for k in keys + ["safe_success_rate_0.5N", "safe_success_rate_1.5N",
                                                                    "peak_tip_force_N", "mean_tip_force_N"]}}}
    if len(L_all) > len(L):
        res["lumen_all_trained_seeds"] = {"seeds": [r["seed"] for r in L_all],
                                          "summary": {k: ci([r.get(k) for r in L_all], "rate" in k) for k in keys}}
    nat = [r["cathsim_native"] for r in C if r.get("cathsim_native")]
    if nat:
        res["cathsim"]["native_metrics_mean"] = {k: sum(x[k] for x in nat) / len(nat) for k in nat[0]}
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))

    def f(c, pct=False, nd=2):
        if c["mean"] is None:
            return "pending"
        s = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.{nd}f}")
        return s(c["mean"]) + (f" [{s(c['lo'])}, {s(c['hi'])}]" if c["lo"] is not None else "")
    LS, CS = res["lumen"]["summary"], res["cathsim"]["summary"]
    md = ["| Metric (mean over training seeds, 95% CI) | Lumen NavTreeBranch | CathSim phantom3/BCA |",
          "|---|---|---|",
          f"| Raw success | {f(LS['success_rate'], True)} | {f(CS['success_rate'], True)} |",
          f"| Safe success at stated threshold | {f(LS['safe_success_rate'], True)} (overlap <= 0.3 mm) | "
          f"{f(CS['safe_success_rate'], True)} (tip force <= 1.0 N) |",
          f"| Safe success, CathSim 0.5 N / 1.5 N | n/a | {f(CS['safe_success_rate_0.5N'], True)} / "
          f"{f(CS['safe_success_rate_1.5N'], True)} |",
          f"| Peak tip contact force (N) | not in N (wall load in sim units) | {f(CS['peak_tip_force_N'])} |",
          f"| Mean tip contact force (N) | not in N | {f(CS['mean_tip_force_N'], nd=3)} |",
          f"| Time to target, simulated (s) | {f(LS['time_to_target_s'])} | {f(CS['time_to_target_s'])} |",
          f"| Training wall-clock for 600k steps (h) | {f(LS['train_hours'])} | {f(CS['train_hours'])} |",
          f"| Training throughput (env-steps/s, 1 env, CPU) | {f(LS['train_env_steps_per_s'], nd=0)} | "
          f"{f(CS['train_env_steps_per_s'], nd=0)} |",
          f"| Evaluation throughput (env-steps/s) | {f(LS['eval_env_steps_per_s'], nd=0)} | "
          f"{f(CS['eval_env_steps_per_s'], nd=0)} |", "",
          "Per training seed (success / safe success):", ""]
    md.append("| Seed | Lumen | CathSim |")
    md.append("|---|---|---|")
    for s in range(6):
        lr = next((r for r in L if r["seed"] == s), None)
        cr = next((r for r in C if r["seed"] == s), None)
        lt = f"{100 * lr['success_rate']:.0f}% / {100 * lr['safe_success_rate']:.0f}%" if lr else "missing"
        ct = (f"{100 * cr['success_rate']:.0f}% / {100 * cr['safe_success_rate']:.0f}%"
              if cr and "success_rate" in cr else (cr or {}).get("status", "missing"))
        md.append(f"| {s} | {lt} | {ct} |")
    if nat:
        n = res["cathsim"]["native_metrics_mean"]
        md += ["", "CathSim on its own published metrics (cathsim/rl/metrics.py, mean over seeds): "
               f"force mean {n['force_mean']:.3f} N, force max {n['force_max']:.3f} N, safety {100 * n['safety']:.1f}%, "
               f"episode length {n['episode_length']:.0f}, path length {n['total_distance_cm']:.1f} cm, "
               f"success {100 * n['success']:.0f}%, SPL {n['spl']:.3f}."]
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
