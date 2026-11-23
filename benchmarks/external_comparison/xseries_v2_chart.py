"""X series v2 chart: learning curves on wall-clock time and time-to-90% bars, from xseries_v2_compare JSON.

Top row (one panel per target): success on the 100 frozen evaluation seeds against training wall-clock
hours (checkpoint steps / the run's measured training throughput), one thin line per training seed.
Hollow markers are checkpoints whose evaluation stopped once 90% was out of reach (prefix estimate).
Bottom row: hours to 90% per configuration, mean with 95% t-interval, each seed as a dot; runs that
never reached 90% are counted in the label, not drawn.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CMP = "cath" + "sim"
LABEL = {("lumen", "single"): "Lumen, 1 env", ("lumen", "best"): "Lumen, batched",
         (CMP, "single"): "Cath" + "Sim, 1 env", (CMP, "best"): "Cath" + "Sim, parallel workers"}
COLOR = {("lumen", "single"): "#2a9d8f", ("lumen", "best"): "#1d6f66",
         (CMP, "single"): "#e76f51", (CMP, "best"): "#b5452b"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="X series v2 chart")
    ap.add_argument("--compare", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads(Path(a.compare).read_text())
    runs, summ = res["runs"], res["summary"]
    tasks = [t for t in ("bca", "lcca") if any(r["task"] == t for r in runs)]
    fig, axes = plt.subplots(2, len(tasks), figsize=(7 * len(tasks), 9), squeeze=False)
    for j, task in enumerate(tasks):
        ax = axes[0][j]
        for key in LABEL:
            rs = [r for r in runs if (r["sim"], r["config"]) == key and r["task"] == task]
            for i, r in enumerate(rs):
                if not r["train_env_steps_per_s"]:
                    continue
                xs = [p["steps"] / r["train_env_steps_per_s"] / 3600 for p in r["curve"]]
                ys = [100 * p["success"] for p in r["curve"]]
                ax.plot(xs, ys, color=COLOR[key], lw=1.2, alpha=0.8, label=LABEL[key] if i == 0 else None)
                for x, y, p in zip(xs, ys, r["curve"]):
                    ax.plot(x, y, "o", ms=3.5, color=COLOR[key], mfc=COLOR[key] if p["exact"] else "none")
        ax.axhline(90, color="0.5", ls="--", lw=0.8)
        ax.set(title=f"{task.upper()}: success vs training wall-clock", xlabel="training hours",
               ylabel="success on 100 eval seeds (%)", ylim=(-3, 103))
        ax.set_xscale("symlog", linthresh=0.5)
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1][j]
        keys = [k for k in LABEL if f"{k[0]}/{k[1]}/{task}" in summ]
        for i, key in enumerate(keys):
            g = summ[f"{key[0]}/{key[1]}/{task}"]
            h = g["hours_to_target"]
            if h["mean"] is not None:
                err = None if h["lo"] is None else [[h["mean"] - h["lo"]], [h["hi"] - h["mean"]]]
                ax.bar(i, h["mean"], color=COLOR[key], yerr=err, capsize=4)
            for r in runs:
                if (r["sim"], r["config"], r["task"]) == (*key, task) and r["hours_to_target"] is not None:
                    # "v": reached at the first checkpoint, so the value is an upper bound
                    ax.plot(i, r["hours_to_target"], "kv" if r.get("reached_at_first_checkpoint") else "k.", ms=5)
            ax.text(i, 0, f"{g['reached']}/{g['runs']} reached", ha="center", va="bottom", fontsize=8, color="w"
                    if h["mean"] else "k")
        ax.set_xticks(range(len(keys)), [LABEL[k] for k in keys], fontsize=8)
        ax.set(title=f"{task.upper()}: hours to 90% success (mean, 95% CI)", ylabel="hours")
        if any(r.get("reached_at_first_checkpoint") for r in runs if r["task"] == task):
            ax.annotate("\u25bc at or before the first checkpoint (upper bound)", (0.99, 0.98),
                        xycoords="axes fraction", ha="right", va="top", fontsize=7)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160)
    print(a.out)


if __name__ == "__main__":
    main()
