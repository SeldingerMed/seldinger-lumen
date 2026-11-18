"""Offline registered planar phantom replay. See docs/PHANTOM_REPLAY.md."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import stat

from lumen.data.schema import Step
from lumen.deployment import BenchTraceMetadata
from lumen.experiments.phantom_capture import admit, synthetic_run, normalized_homography
from lumen.experiments.phantom_metrics import compare
from lumen.experiments.phantom_solver import convert_samples, replay

METHOD = "lumen-phantom/1:left-fork:mean-wire-reset-translation"
TOOLS = ("wire", "catheter")
MAX_BYTES = 8 * 1024 * 1024


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read(path):
    def unique(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON key")
        return result

    if path.is_symlink() or not path.is_file():
        raise ValueError("expected local regular input file")
    with path.open("rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise ValueError("expected regular input file")
        data = f.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("input exceeds byte limit")
    obj = json.loads(data, object_pairs_hook=unique)
    digest(obj)  # refuse nonfinite nested data, including command-only fields
    return obj


def write(path, obj):
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8"
    ) as f:
        json.dump(obj, f, allow_nan=False, indent=1)
        f.write("\n")


def plan_at(root):
    plan = read(root / "plan.json")
    if not isinstance(plan, dict) or plan.get("method") != METHOD:
        raise ValueError("unsupported frozen method")
    seen = set()
    for split in ("calibration", "holdout"):
        ids = plan.get(split)
        if not isinstance(ids, list) or not 1 <= len(ids) <= 32:
            raise ValueError("reserve 1..32 complete runs per split")
        for run in ids:
            if not isinstance(run, str) or not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run):
                raise ValueError("invalid run ID")
            if run.casefold() in {"complete", "plan", "fit"}:
                raise ValueError("reserved run ID")
            if run.casefold() in seen:
                raise ValueError("duplicate run ID")
            seen.add(run.casefold())
    return plan


def trial(root, run_id):
    data = read(root / (run_id + ".json"))
    rows = admit(data)
    meta = data["metadata"]
    if meta["run_id"] != run_id:
        raise ValueError("run ID does not match assignment")
    # Canonical signed numeric samples; ignore headers, commands, tip labels,
    # timestamps and -0.0. Relabelling a replay cannot create a new maneuver.
    motion = [
        [
            [float(v * s) or 0.0 for v, s in zip(r[k], meta["signs"][k])]
            if r[k] is not None
            else None
            for k in TOOLS
        ]
        for r in rows
    ]
    binding = {
        k: meta[k]
        for k in ("phantom_id", "calibration_id", "homography", "signs", "units", "provenance")
    }
    binding["tool_ids"] = meta["tool_ids"]
    binding["homography"] = normalized_homography(meta["homography"])
    return data, {
        "run_id": run_id,
        "recording_id": meta["recording_id"],
        "motion": digest(motion),
        "source": digest(data),
        "setup": digest(binding),
    }


def distinct(records):
    for field in ("recording_id", "motion"):
        if len({r[field] for r in records}) != len(records):
            raise ValueError("duplicate " + field + " across complete runs")
    if len({r["setup"] for r in records}) != 1:
        raise ValueError("incompatible calibration, provenance or phantom")


def simulate(run):
    return replay(run["rows"], run["metadata"]["signs"])


def joined(run, simulation, translation):
    rows = []
    for raw, step in zip(run["rows"], simulation, strict=True):
        sim = {k: step[k] for k in TOOLS}
        row = dict(
            raw,
            raw_sim_xy=sim,
            sim_xy={
                k: [x + b for x, b in zip(p, translation)] if p is not None else None
                for k, p in sim.items()
            },
        )
        rows.append(row)
    # Compare shapes/branch regions in model coordinates; translation registers
    # the camera to the model, not the geometry to a different decision plane.
    canonical = []
    for row in rows:
        m = {
            k: ([x - b for x, b in zip(p, translation)] if p is not None else None)
            for k, p in row["measured_xy"].items()
        }
        c: dict = dict(row, measured_xy=m, sim_xy=row["raw_sim_xy"])
        for k in TOOLS:
            if c[k] is not None:
                c[k] = [x * s for x, s in zip(c[k], run["metadata"]["signs"][k])]
        canonical.append(c)
    return rows, compare(canonical)


def calibrate(root):
    plan = plan_at(root)
    records, offsets, reports = [], [], {}
    for run_id in plan["calibration"]:  # never open a holdout during fitting
        run, rec = trial(root, run_id)
        records.append(rec)
        distinct(records)
        sim = simulate(run)
        tip = run["rows"][0]["measured_xy"]["wire"]
        if tip is None:
            raise ValueError("missing visible calibration reset wire tip")
        delta = [x - y for x, y in zip(tip, sim[0]["wire"])]
        if math.hypot(*delta) > 5:
            raise ValueError("reset outside 5 mm registration budget")
        offsets.append(delta)
        metrics = joined(run, sim, [0, 0])[1]
        reports[run_id] = {k: v for k, v in metrics.items() if k != "discrepancies"}
    fit = dict(
        method=METHOD,
        plan=digest(plan),
        records=records,
        translation_mm=[sum(axis) / len(offsets) for axis in zip(*offsets)],
        calibration_metrics_before_fit=reports,
    )
    write(root / "fit.json", fit)
    return digest(fit)


def render(report):
    esc = lambda data: html.escape(json.dumps(data, ensure_ascii=True, allow_nan=False))
    summary = report["metrics"]["first_consequential_discrepancy"]
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Phantom replay</title>',
        "<h1>Offline phantom replay: " + html.escape(report["run_id"]) + "</h1>",
        "<p>Evidence: "
        + html.escape(report["provenance"])
        + ". First consequential discrepancy: "
        + (esc(summary) if summary else "none identifiable from matched observations")
        + "</p>",
        "<p>Coordinates in registered camera-plane mm. Null means not measured or not modeled, not zero.</p>",
        "<table><caption>Synchronized replay</caption><tr><th>Time (s)</th><th>Tool</th>"
        "<th>Measured XY</th><th>Simulated XY</th><th>Confidence</th></tr>",
    ]
    for r in report["rows"]:
        for k in TOOLS:
            parts.append(
                "<tr>"
                + "".join(
                    "<td>" + esc(v) + "</td>"
                    for v in (r["t_s"], k, r["measured_xy"][k], r["sim_xy"][k], r["confidence"][k])
                )
                + "</tr>"
            )
    return (
        "".join(parts)
        + "</table><h2>Comparisons and coverage</h2><pre>"
        + html.escape(json.dumps(report["metrics"], indent=2))
        + "</pre>"
    )


def evaluate(root, frozen_hash):
    plan = plan_at(root)
    fit = read(root / "fit.json")
    if digest(fit) != frozen_hash or fit["plan"] != digest(plan) or fit["method"] != METHOD:
        raise ValueError("frozen fit/plan mismatch")
    records = list(fit["records"])
    for run_id, expected in zip(plan["calibration"], records, strict=True):
        if trial(root, run_id)[1] != expected:
            raise ValueError("calibration source changed after freeze")
    holdouts = []
    for run_id in plan["holdout"]:
        run, rec = trial(root, run_id)
        records.append(rec)
        distinct(records)
        holdouts.append((run, rec))
    out = root / "evaluation"
    out.mkdir(mode=0o700)  # fresh only; incomplete folders lack complete.json
    for run, rec in holdouts:
        rows, metrics = joined(run, simulate(run), fit["translation_mm"])
        report = dict(
            run_id=rec["run_id"],
            source=rec,
            provenance=run["metadata"]["provenance"],
            metadata=run["metadata"],
            fit_sha256=frozen_hash,
            translation_mm=fit["translation_mm"],
            rows=rows,
            metrics=metrics,
        )
        meta = run["metadata"]
        inlet = meta["inlet_sensor"] if any(r["inlet_force_n"] is not None for r in rows) else {}
        report["bench_trace_metadata"] = BenchTraceMetadata(
            source="bench",
            force_units="N" if inlet else "not_measured",
            torque_units="not_measured",
            device_id=meta["tool_ids"]["wire"],
            phantom_id=meta["phantom_id"],
            actuation_protocol=METHOD,
            calibration_reference=inlet.get("calibration_id", meta["calibration_id"]),
            provenance=meta["provenance"],
        ).to_dict()
        deltas = [()] + convert_samples(rows, meta["signs"])
        report["data_steps"] = [
            Step(
                t=r["t_s"],
                action=dict(zip(("insertion", "twist", "insertion_cath", "twist_cath"), delta)),
                kinematics={"measured_xy": r["measured_xy"], "registered_sim_xy": r["sim_xy"]},
                force=r["inlet_force_n"],
            ).to_dict()
            for r, delta in zip(rows, deltas, strict=True)
        ]
        write(out / (rec["run_id"] + ".json"), report)
        with (out / (rec["run_id"] + ".html")).open("x") as f:
            f.write(render(report))
    write(out / "complete.json", dict(plan=digest(plan), fit=frozen_hash, holdout=plan["holdout"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("demo", "calibrate", "evaluate"):
        p = sub.add_parser(mode)
        p.add_argument("workspace", type=Path)
        if mode == "evaluate":
            p.add_argument("--fit-sha256", required=True)
    args = parser.parse_args(argv)
    root = args.workspace.resolve()
    try:
        if args.mode == "demo":
            root.mkdir(mode=0o700)
            write(
                root / "plan.json", dict(method=METHOD, calibration=["cal01"], holdout=["hold01"])
            )
            for label, variant in [("cal01", "straight"), ("hold01", "partial")]:
                write(root / (label + ".json"), synthetic_run(label, "synthetic-" + label, variant))
        frozen = getattr(args, "fit_sha256", "")
        if args.mode != "evaluate":
            frozen = calibrate(root)
            print("Fit SHA-256:", frozen)
        if args.mode != "calibrate":
            evaluate(root, frozen)
            print(root / "evaluation" / "complete.json")
    except (OSError, ValueError, RecursionError) as exc:
        parser.exit(2, f"phantom replay refused: {exc}\n")


if __name__ == "__main__":
    main()
