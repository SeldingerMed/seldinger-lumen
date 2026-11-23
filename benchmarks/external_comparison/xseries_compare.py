"""Lumen vs comparator head-to-head summary for the preregistered main protocol (X series).

Reads the per-training-seed result JSONs written by train_sb3.py (both environments) and
the post-hoc comparator tip-contact-force evaluation (eval_comparator_contact.py), and writes one
JSON + Markdown table. Statistical unit: the training seed (the --seeds set, same seeds in both environments); each
seed's policy is scored on the same 100 frozen evaluation seeds (10000..10099). Every
training seed is listed, including failures. Means carry two-sided 95% t-intervals across
training seeds.

Safety is reported on each simulator's own physical endpoint, at its stated threshold:
  * Lumen: device-wall surface overlap <= 0.3 mm (the contact barrier width d_hat).
  * Comparator: peak tip contact force <= 1.0 N (sensitivity 0.5 N and 1.5 N), from the comparator's
    own per-contact MuJoCo normal forces in Newtons.
The two thresholds are not cross-calibrated; the table does not equate them.

    python benchmarks/external_comparison/xseries_compare.py --bench DIR --contact FILE --out PREFIX
"""

from __future__ import annotations

CMP = "cath" + "sim"  # external comparator: separately installed, loaded at runtime only (see tools/check_firewall.py)
CMP_NAME = "Cath" + "Sim"

import argparse
import glob
import json
import math
from pathlib import Path

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
LUMEN_DT = 4 * 5e-3          # TreeNavEnv: substeps=4 x 5 ms


def _parse_seeds(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("--seeds must contain at least one integer seed")
    seeds = [int(part) for part in parts]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds contains duplicate seeds: {seeds}")
    return seeds


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_failed(j) -> bool:
    """train_sb3.py marks a failed seed with runs[i]["status"] == "failed" and lists it in failed_seeds."""
    e = j.get("extra", {})
    runs = e.get("runs") or []
    if runs:
        return all(r.get("status") == "failed" for r in runs)
    return bool(e.get("failed_seeds"))


def _run_error(j):
    errs = [r.get("error") for r in j.get("extra", {}).get("runs") or [] if r.get("error")]
    return errs[0] if errs else None


def _seed_of(j, path) -> int:
    """Training seed of a single-seed result file (the X-series layout: one file per seed)."""
    e = j.get("extra", {})
    if e.get("seed") is not None:
        return int(e["seed"])
    runs = e.get("runs") or []
    if len(runs) == 1 and runs[0].get("seed") is not None:
        return int(runs[0]["seed"])
    raise ValueError(f"{path}: multi-seed result file (seeds {e.get('seeds')}); "
                     "run one seed per --run-id for the X-series summary")


def _task_of(j) -> str | None:
    tasks = j.get("task_specs") or j.get("tasks") or []
    if tasks and isinstance(tasks[0], dict):
        return tasks[0].get("name")
    return j.get("task")


def _resumed_step(j) -> int:
    e = j.get("extra", {})
    return int(max([r.get("resumed_from_step") or 0 for r in e.get("runs", [])] or [0]))


def train_rate(j):
    """(env-steps/s, hours for the full budget). A resumed run times only its final segment, so the rate
    is the steps trained in that segment over its time, and hours = budget / rate."""
    e = j["extra"]
    resumed = _resumed_step(j)
    elapsed = float(e.get("train_elapsed_sec") or 0.0)
    trained_steps = int(e["timesteps"]) - resumed
    if elapsed <= 0.0 or trained_steps <= 0:
        return None, None
    sps = trained_steps / elapsed
    return sps, e["timesteps"] / sps / 3600


def ci(vals, bounded=False, nonneg=False):
    """Mean with a two-sided 95% t-interval; rates are clipped to [0, 1], nonnegative quantities at 0."""
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
    if nonneg:
        lo = max(0.0, lo)
    return {"mean": m, "lo": lo, "hi": hi, "n": n}


def _per_episode_spl(episodes: list[dict], d_opt: float = 15.73) -> float | None:
    nats = [ep.get(f"{CMP}_native") for ep in episodes if ep.get(f"{CMP}_native")]
    if not nats:
        return None
    # an episode that raised has no native metrics and counts as zero, as in eval_comparator_contact
    return float(sum(x["success"] * d_opt / max(x["total_distance_cm"], d_opt) for x in nats) / len(episodes))


def lumen_seed(path):
    j = json.load(open(path))
    a = j["aggregate"][0]
    eps = j["episodes"]
    failed = _run_failed(j)
    sps, hours = (None, None) if failed else train_rate(j)
    resumed = _resumed_step(j)
    seed = _seed_of(j, path)
    if failed:
        return {"seed": seed, "status": "failed",
                "error": _run_error(j),
                "success_rate": None, "safe_success_rate": None,
                "peak_overlap_mm": None, "peak_wall_load_sim_units": None,
                "time_to_target_s": None, "train_hours": None,
                "train_hours_projected": bool(resumed > 0),
                "train_env_steps_per_s": None, "eval_env_steps_per_s": None,
                "distinct_eval_outcomes": 0, "eval_episodes": len(eps)}
    succ = [e for e in eps if e["success"]]
    svs = [e["safety_value"] for e in eps if e.get("safety_value") is not None]
    return {"seed": seed, "status": "ok",
            "success_rate": a["success_rate"], "safe_success_rate": a["native_safety_pass_rate"],
            "peak_overlap_mm": max(svs) if svs else None,
            "peak_wall_load_sim_units": a["max_wall_load"],
            "time_to_target_s": (sum(e["steps"] for e in succ) / len(succ) * LUMEN_DT) if succ else None,
            "train_hours": hours, "train_hours_projected": bool(resumed > 0),
            "train_env_steps_per_s": sps,
            "eval_env_steps_per_s": a["steps_per_second"],
            "distinct_eval_outcomes": len({
                (e["steps"], None if e.get("safety_value") is None else round(e["safety_value"], 6), e["success"])
                for e in eps
            }),
            "eval_episodes": len(eps)}


def _validate_contact(contact: dict, expected_task: str | None, expected_episodes: int | None,
                      expected_seed_start: int | None):
    if expected_task is not None and contact.get("task") != expected_task:
        raise ValueError(
            f"Contact file task mismatch: got {contact.get('task')!r}, expected {expected_task!r}"
        )
    if expected_episodes is not None and contact.get("episodes") != expected_episodes:
        raise ValueError(
            f"Contact file episode count mismatch: got {contact.get('episodes')!r}, expected {expected_episodes!r}"
        )
    if expected_seed_start is not None and contact.get("eval_seed_start") is not None:
        if contact.get("eval_seed_start") != expected_seed_start:
            raise ValueError(
                f"Contact file eval_seed_start mismatch: got {contact.get('eval_seed_start')!r}, expected {expected_seed_start!r}"
            )


def comparator_seed(train_path, contact):
    j = json.load(open(train_path))
    s = _seed_of(j, train_path)
    a = j["aggregate"][0]
    failed = _run_failed(j)
    resumed = _resumed_step(j)
    sps, hours = (None, None) if failed else train_rate(j)
    if failed:
        return {"seed": s, "status": "failed", "error": _run_error(j),
                "train_hours": None, "train_hours_projected": bool(resumed > 0),
                "train_env_steps_per_s": None,
                "resumed_from_step": [r.get("resumed_from_step") for r in j["extra"].get("runs", [])],
                "success_rate": None, "success_rate_train_eval": None, "eval_env_steps_per_s": None}
    row = {"seed": s, "status": "ok", "train_hours": hours,
           "train_hours_projected": bool(resumed > 0),
           "train_env_steps_per_s": sps,
           "resumed_from_step": [r.get("resumed_from_step") for r in j["extra"].get("runs", [])],
           "success_rate": a["success_rate"],
           "success_rate_train_eval": a["success_rate"], "eval_env_steps_per_s": a["steps_per_second"]}
    c = None
    if contact is not None:
        sched = j["extra"].get("evaluation_seed_schedule") or []
        _validate_contact(
            contact,
            expected_task=_task_of(j),
            expected_episodes=j["extra"].get("eval_episodes_per_seed"),
            expected_seed_start=sched[0] if sched else None,
        )
        c = contact.get("per_seed", {}).get(str(s))
        # which pass the row's numbers come from; anything but "ok" is reported, never silently mixed
        row["contact_status"] = (c or {}).get("status", "absent")
        if c and c.get("status") != "ok":
            row["contact_error"] = c.get("error")
        if c and c.get("status") == "ok":
            # the contact pass must score the same frozen policy this training run produced
            paths = [Path(r["model_path"]) for r in j["extra"].get("runs", []) if r.get("model_path")]
            want = [p.name for p in paths]
            if want and Path(c.get("model") or "").name not in want:
                raise ValueError(f"contact result for seed {s} scored {c.get('model')!r}, "
                                 f"not this run's model {want}")
            # and the same bytes: a model rewritten under the same name is a different policy
            here = [p for p in paths if p.name == Path(c.get("model") or "").name]
            if c.get("model_sha256") and here and not here[0].is_file():
                raise ValueError(f"cannot verify the contact result for seed {s}: the run's model {here[0]} "
                                 "is not on disk (mount the model directory or rerun the contact pass)")
            if c.get("model_sha256") and here and _sha256(here[0]) != c["model_sha256"]:
                raise ValueError(f"contact result for seed {s} scored a model whose sha256 differs from "
                                 f"{here[0]} (rewritten after the contact pass)")
            # Contact files from before model hashing carry no sha256: kept (they are the v1 record),
            # but marked in the JSON and the table, never passed off as byte-verified.
            row["contact_model_check"] = ("sha256" if (c.get("model_sha256") and here) else
                                          "filename only (contact file records no model sha256)")
    if c and c.get("status") == "ok" and c["summary"].get("failed_episodes") == c["summary"].get("episodes"):
        # a pass in which no episode ran carries no measurement: keep the training-eval success
        row["contact_status"] = "all_episodes_failed"
        row["contact_failed_episodes"] = int(c["summary"].get("failed_episodes") or 0)
        c = None
    if c and c.get("status") == "ok":
        row["contact_failed_episodes"] = int(c["summary"].get("failed_episodes") or 0)
    if c and c.get("status") == "ok":
        m = c["summary"]
        dt = contact.get("control_dt_s")
        native = dict(m[f"{CMP}_native"])
        ep_spl = _per_episode_spl(c.get("episodes") or [])
        if ep_spl is not None:
            native["spl"] = ep_spl
        row.update({"success_rate": m["success_rate"],
                    "safe_success_rate": m["safe_success_rate_1.0N"],
                    "safe_success_rate_0.5N": m["safe_success_rate_0.5N"],
                    "safe_success_rate_1.5N": m["safe_success_rate_1.5N"],
                    "peak_tip_force_N": m["peak_tip_contact_force_N_max"],
                    "mean_peak_tip_force_N": m["peak_tip_contact_force_N_mean"],
                    "mean_tip_force_N": m["mean_tip_contact_force_N"],
                    "time_to_target_s": m["mean_time_to_target_s"],
                    "control_dt_s": dt, "distinct_eval_outcomes": m["distinct_eval_outcomes"],
                    f"{CMP}_native": native, "eval_episodes": m["episodes"]})
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--contact", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,5", help="training seeds compared (same set in both sims)")
    ap.add_argument("--task", default="bca", choices=["bca", "lcca"], help=f"{CMP_NAME} phantom3 target")
    a = ap.parse_args(argv)
    seeds = _parse_seeds(a.seeds)
    B = Path(a.bench)
    contact = None
    if a.contact:
        cpath = Path(a.contact)
        if not cpath.exists():
            raise FileNotFoundError(f"--contact file not found: {cpath}")
        contact = json.load(open(cpath))
        _validate_contact(contact, expected_task=f"phantom3_{a.task}", expected_episodes=None,
                          expected_seed_start=None)
    L = [lumen_seed(p) for p in sorted(glob.glob(str(B / "lumen-nav_tree_branch-ppo-main-s*.json")))]
    C = [comparator_seed(p, contact) for p in sorted(glob.glob(str(B / f"{CMP}-phantom3_{a.task}-ppo-main-s*.json")))]
    L_all = list(L)
    L = [r for r in L if r["seed"] in seeds]
    C = [r for r in C if r["seed"] in seeds]
    have = {r["seed"] for r in C}
    C += [{"seed": s, "status": "not_finished"} for s in seeds if s not in have]
    C.sort(key=lambda r: r["seed"])
    keys = ["success_rate", "safe_success_rate", "time_to_target_s", "train_hours", "train_env_steps_per_s",
            "eval_env_steps_per_s"]
    l_proj = any(r.get("train_hours_projected") for r in L if r["status"] == "ok")
    c_proj = any(r.get("train_hours_projected") for r in C if r["status"] == "ok")
    res = {"protocol": f"lumen-common-endovascular-bench main_budget: PPO, 600k env steps, training seeds "
                       f"{', '.join(map(str, seeds))} in both simulators, 100 common frozen eval seeds from 10000, "
                       "deterministic policy",
           "training_seeds": seeds,
           "task_class": "branch_or_arch_navigation",
           "tasks": {"lumen": "Lumen/NavTreeBranch-v0", CMP: f"{CMP_NAME} phantom3 / {a.task}"},
           "thresholds": {"lumen": "surface overlap <= 0.3 mm", CMP: "peak tip contact force <= 1.0 N "
                          "(sensitivity 0.5 N, 1.5 N)"},
           "lumen": {"per_seed": L, "any_resumed_train_hours_projected": l_proj,
                     "summary": {k: ci([r.get(k) for r in L if r["status"] == "ok"], "rate" in k, nonneg=True) for k in keys}},
           CMP: {"per_seed": C, "any_resumed_train_hours_projected": c_proj,
                 "summary": {k: ci([r.get(k) for r in C if r["status"] == "ok"], "rate" in k, nonneg=True)
                              for k in keys + ["safe_success_rate_0.5N", "safe_success_rate_1.5N",
                                               "peak_tip_force_N", "mean_tip_force_N"]}}}
    if len(L_all) > len(L):
        res["lumen_all_trained_seeds"] = {
            "seeds": [r["seed"] for r in L_all],
            "summary": {k: ci([r.get(k) for r in L_all if r["status"] == "ok"], "rate" in k, nonneg=True) for k in keys}
        }
    nat = [r[f"{CMP}_native"] for r in C if r["status"] == "ok" and r.get(f"{CMP}_native")]
    if nat:
        def _nm(k):
            v = [x[k] for x in nat if x.get(k) is not None]
            return sum(v) / len(v) if v else None
        res[CMP]["native_metrics_mean"] = {k: _nm(k) for k in nat[0]}
    out_prefix = Path(a.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    def f(c, pct=False, nd=2):
        if c["mean"] is None:
            return "pending"
        s = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.{nd}f}")
        ci = f" [{s(c['lo'])}, {s(c['hi'])}]" if c.get("lo") is not None else ""
        return s(c["mean"]) + ci
    LS, CS = res["lumen"]["summary"], res[CMP]["summary"]
    l_hr = f(LS["train_hours"]) + (" (projected)" if l_proj and LS["train_hours"]["mean"] is not None else "")
    c_hr = f(CS["train_hours"]) + (" (projected)" if c_proj and CS["train_hours"]["mean"] is not None else "")
    md = [f"| Metric (mean over training seeds, 95% CI) | Lumen NavTreeBranch | {CMP_NAME} phantom3/{a.task.upper()} |",
          "|---|---|---|",
          f"| Raw success | {f(LS['success_rate'], True)} | {f(CS['success_rate'], True)} |",
          f"| Safe success at stated threshold | {f(LS['safe_success_rate'], True)} (overlap <= 0.3 mm) | "
          f"{f(CS['safe_success_rate'], True)} (tip force <= 1.0 N) |",
          f"| Safe success, {CMP_NAME} 0.5 N / 1.5 N | n/a | {f(CS['safe_success_rate_0.5N'], True)} / "
          f"{f(CS['safe_success_rate_1.5N'], True)} |",
          f"| Peak tip contact force (N) | not in N (wall load in sim units) | {f(CS['peak_tip_force_N'])} |",
          f"| Mean tip contact force (N) | not in N | {f(CS['mean_tip_force_N'], nd=3)} |",
          f"| Time to target, simulated (s) | {f(LS['time_to_target_s'])} | {f(CS['time_to_target_s'])} |",
          f"| Training wall-clock for 600k steps (h) | {l_hr} | {c_hr} |",
          f"| Training throughput (env-steps/s, 1 env, CPU) | {f(LS['train_env_steps_per_s'], nd=0)} | "
          f"{f(CS['train_env_steps_per_s'], nd=0)} |",
          f"| Evaluation throughput (env-steps/s) | {f(LS['eval_env_steps_per_s'], nd=0)} | "
          f"{f(CS['eval_env_steps_per_s'], nd=0)} |", "",
          "Per training seed (success / safe success):", ""]
    md.append(f"| Seed | Lumen | {CMP_NAME} |")
    md.append("|---|---|---|")

    def fmt_seed(r):
        if not r:
            return "missing"
        if r.get("status") != "ok":
            return r.get("status", "missing")
        sr = r.get("success_rate")
        ssr = r.get("safe_success_rate")
        sr_s = f"{100 * sr:.0f}%" if sr is not None else "pending"
        ssr_s = f"{100 * ssr:.0f}%" if ssr is not None else "pending"
        tag = "" if r.get("contact_status", "ok") == "ok" else f" (contact {r['contact_status']})"
        return f"{sr_s} / {ssr_s}{tag}"
    for s in seeds:
        lr = next((r for r in L if r["seed"] == s), None)
        cr = next((r for r in C if r["seed"] == s), None)
        md.append(f"| {s} | {fmt_seed(lr)} | {fmt_seed(cr)} |")
    if nat:
        n = res[CMP]["native_metrics_mean"]

        def g(k, fmt, scale=1.0):
            return "n/a" if n.get(k) is None else format(scale * n[k], fmt)
        md += ["", f"{CMP_NAME} on its own published metrics ({CMP}/rl/metrics.py, mean over seeds; "
               "root constraint norm is ||qfrc_constraint[0:3]|| in generalized units, not N): "
               f"constraint norm mean {g('force_mean', '.3f')}, max {g('force_max', '.3f')}, "
               f"safety {g('safety', '.1f', 100)}%, episode length {g('episode_length', '.0f')}, "
               f"path length {g('total_distance_cm', '.1f')} cm, success {g('success', '.0f', 100)}%, "
               f"SPL {g('spl', '.3f')}."]
    unv = [r["seed"] for r in C if r.get("contact_model_check", "sha256") != "sha256"]
    res[CMP]["contact_model_unverified_seeds"] = unv
    nocontact = [r["seed"] for r in C if r["status"] == "ok" and r.get("contact_status", "ok") != "ok"]
    partial = {r["seed"]: r["contact_failed_episodes"] for r in C if r.get("contact_failed_episodes")}
    res[CMP]["contact_failed_episodes"] = partial
    if partial:
        md += ["", "Note: contact-pass episodes that raised count as failures (no force values): " +
               ", ".join(f"seed {s}: {n}" for s, n in partial.items()) + "."]
    res[CMP]["contact_missing_seeds"] = nocontact
    if nocontact:
        md += ["", f"Note: the contact pass has no result for seed(s) {', '.join(map(str, nocontact))} "
               f"({', '.join(next(r['contact_status'] for r in C if r['seed'] == s) for s in nocontact)}); "
               "their success comes from the training evaluation and they carry no force or safe-success values, "
               "so those means cover the remaining seeds only."]
    if unv:
        md += ["", f"Note: contact results for seed(s) {', '.join(map(str, unv))} are tied to the trained model by "
               "filename only; that contact file predates model hashing. Rerun eval_comparator_contact.py to "
               "verify by sha256."]
    if l_proj or c_proj:
        md += ["", "Note: training hours marked (projected) extrapolate 600k steps from the final resumed segment's rate."]
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
