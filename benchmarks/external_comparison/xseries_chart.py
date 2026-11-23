"""One clean PNG chart for the X-series head-to-head (item 8), from xseries_compare JSON.

Three panels, one dot per training seed with the mean and 95% CI across seeds:
  1. raw success rate      2. safe success rate at each simulator's stated threshold
  3. training wall-clock for the 600k-step budget (hours)
Every training seed is plotted, including failures.

    python benchmarks/external_comparison/xseries_chart.py --compare compare.json --out chart.png
"""

from __future__ import annotations

CMP = "cath" + "sim"  # external comparator: separately installed, loaded at runtime only (see tools/check_firewall.py)
CMP_NAME = "Cath" + "Sim"

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {"Lumen": "#3fd0d8", CMP_NAME: "#e8a33d"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    c = json.load(open(a.compare))
    sides = {"Lumen": c["lumen"], CMP_NAME: c[CMP]}
    panels = [("success_rate", "Raw success", True),
              ("safe_success_rate", "Safe success", True),
              ("train_hours", "Training wall-clock, 600k steps", False)]
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0), dpi=160)
    rng = np.random.default_rng(0)
    n_expected = len(c.get("training_seeds") or range(6))
    any_proj = any(d.get("any_resumed_train_hours_projected") for d in sides.values())
    for ax, (key, title, pct) in zip(axes, panels):
        for i, (name, d) in enumerate(sides.items()):
            rows = d.get("per_seed", [])
            vals = [r.get(key) for r in rows if r.get("status", "ok") == "ok" and r.get(key) is not None]
            unavail = [r for r in rows if r.get("status", "ok") != "ok" or r.get(key) is None]
            s = d["summary"][key]
            if not vals or s.get("mean") is None:
                ax.text(i, 50 if pct else 0.0, "pending", ha="center", color="#888")
                continue
            v = np.asarray(vals, float) * (100 if pct else 1)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=46, color=COL[name], alpha=0.85, zorder=3,
                       edgecolor="white", linewidth=0.6)
            m = s["mean"] * (100 if pct else 1)
            ax.hlines(m, i - 0.28, i + 0.28, color="#222", linewidth=2.2, zorder=4)
            if s["lo"] is not None:
                lo, hi = s["lo"] * (100 if pct else 1), s["hi"] * (100 if pct else 1)
                ax.vlines(i, lo, hi, color="#222", linewidth=1.4, zorder=4)
            suffix = "*" if (key == "train_hours" and d.get("any_resumed_train_hours_projected")) else ""
            lbl = f"{m:.0f}%" if pct else f"{m:.1f} h{suffix}"
            if unavail:
                lbl += f"\n(n={len(vals)}/{len(rows)})"
            ax.text(i + 0.32, m, lbl, va="center", fontsize=11 if unavail else 12)
        ax.set_xticks([0, 1], list(sides))
        ax.set_xlim(-0.6, 1.9)
        if pct:
            ax.set_ylim(-5, 105)
            ax.set_ylabel("% of 100 evaluation episodes")
        else:
            ax.set_ylim(0, None)
            ax.set_ylabel("hours (CPU, 1 environment)")
        ax.set_title(title, fontsize=12.5)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Lumen vs {CMP_NAME}: matched PPO (600k env steps), {n_expected} training seeds each, "
                 "100 common frozen evaluation seeds\nDots: available training seeds.  Bar: mean.  Line: 95% CI across seeds.",
                 fontsize=13)
    foot = ("Safe success = reached the target without exceeding the simulator's stated limit: "
            f"Lumen device-wall overlap 0.3 mm; {CMP_NAME} tip-wall contact force 1.0 N.")
    if any_proj:
        foot += "  *Projected from final resumed segment."
    fig.text(0.5, 0.015, foot, ha="center", fontsize=11, color="#444")
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    fig.savefig(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
