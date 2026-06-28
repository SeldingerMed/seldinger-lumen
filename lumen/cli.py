"""Installed command-line entry points for the common Lumen workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from lumen.hardware import describe


def _command_table():
    return {
        "hardware": ("Print backend hardware/software status.", hardware_main),
        "benchmark": ("Run the canonical navigation benchmark.", benchmark_main),
        "render-fluoro": ("Render the canonical synthetic fluoroscopy demo.", render_fluoro_main),
        "capture": ("Capture the canonical procedural case-bundle corpus.", capture_main),
        "replay": ("Summarize and replay a case-bundle corpus.", replay_main),
        "validate": ("Validate a case-bundle corpus before training.", validate_main),
        "index": ("Write a JSONL dataloader index for a corpus.", index_main),
        "calibrate": ("Run the wall-probe calibration identifiability demo.", calibrate_main),
    }


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = _command_table()
    parser = argparse.ArgumentParser(
        prog="lumen",
        description="Lumen first-run workflows for endovascular RL/CV datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    for name, (help_text, _) in commands.items():
        subparsers.add_parser(name, help=help_text)
    if not argv:
        parser.print_help()
        return
    if argv[0] in commands:
        commands[argv[0]][1](argv[1:], prog=f"lumen {argv[0]}")
        return
    parser.parse_args(argv)


def hardware_main(argv=None, prog=None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog, description="Print Lumen backend hardware/software status.")
    parser.parse_args(argv)
    print(json.dumps(describe(), indent=2))


def benchmark_main(argv=None, prog=None) -> None:
    from lumen.bench import (SAFETY_MAX_PEN, evaluate_policy, forward_policy, leaderboard,
                             scorecard_rejections)

    parser = argparse.ArgumentParser(
        prog=prog, description="Run the canonical Lumen navigation benchmark.")
    parser.add_argument("results_dir", nargs="?", default="bench_results")
    args = parser.parse_args(argv)
    os.makedirs(args.results_dir, exist_ok=True)

    sc = evaluate_policy(forward_policy, "forward-baseline", notes={
        "policy": "lumen.bench.forward_policy",
        "command": "lumen benchmark",
        "safety_max_pen": SAFETY_MAX_PEN,
    })
    sc.save(os.path.join(args.results_dir, "forward-baseline.json"))

    print(f"suite {sc.suite_version}   submission: {sc.name}")
    print(f"{'task':18} {'tier':7} {'safe':>8} {'unsafe':>8} {'success':>8} "
          f"{'mean_steps':>11} {'max_pen':>9}")
    for t in sc.per_task:
        steps = "-" if t["mean_steps"] is None else f"{t['mean_steps']:.1f}"
        print(f"{t['name']:18} {t['tier']:7} {t['safe_success_rate']:>8.2f} "
              f"{t['unsafe_success_rate']:>8.2f} {t['success_rate']:>8.2f} "
              f"{steps:>11} {t['max_pen']:>9.3f}")
    o = sc.overall
    print(f"\noverall: safe={o['safe_success_rate']:.2f}  "
          f"unsafe={o['unsafe_success_rate']:.2f}  success={o['success_rate']:.2f}  "
          f"worst max_pen={o['max_pen']:.3f}  mean_return={o['mean_return']:.1f}")

    print(f"\nleaderboard ({args.results_dir}):")
    for rank, c in enumerate(leaderboard(args.results_dir), 1):
        print(f"  {rank}. {c.name:24} safe={c.overall.get('safe_success_rate', 0.0):.2f}  "
              f"unsafe={c.overall.get('unsafe_success_rate', 0.0):.2f}  "
              f"success={c.overall['success_rate']:.2f}  "
              f"max_pen={c.overall['max_pen']:.3f}  return={c.overall['mean_return']:.1f}")
    skipped = scorecard_rejections(args.results_dir)
    if skipped:
        print("\nskipped scorecards:")
        for item in skipped:
            print(f"  {item['path']}: {item['error']}")


def render_fluoro_main(argv=None, prog=None) -> None:
    from lumen.workflows import render_fluoro_example

    parser = argparse.ArgumentParser(
        prog=prog, description="Render the canonical Lumen fluoro demo.")
    parser.add_argument("out_png", nargs="?", default="fluoro.png")
    args = parser.parse_args(argv)
    render_fluoro_example(args.out_png)


def capture_main(argv=None, prog=None) -> None:
    from lumen.workflows import capture_examples

    parser = argparse.ArgumentParser(
        prog=prog, description="Capture the canonical procedural Lumen case corpus.")
    parser.add_argument("out_dir", nargs="?", default="episodes")
    args = parser.parse_args(argv)
    capture_examples(args.out_dir)


def _clinical_flags(ep):
    metrics = ep.outcome.metrics if isinstance(ep.outcome.metrics, dict) else {}
    tip = metrics.get("tip_target") if isinstance(metrics.get("tip_target"), dict) else {}
    wall = metrics.get("wall_safety") if isinstance(metrics.get("wall_safety"), dict) else {}
    branch = metrics.get("branch_choice") if isinstance(metrics.get("branch_choice"), dict) else {}
    parts = []
    if "success" in tip:
        parts.append(f"tip_target={tip['success']!s}")
    if "perforation_risk" in wall:
        parts.append(f"wall_risk={wall['perforation_risk']!s}")
    if branch.get("correct") is not None:
        parts.append(f"branch={branch['correct']!s}")
    return "  ".join(parts)


def _annotation_flags(ep):
    from lumen.data import annotation_coverage

    cov = annotation_coverage(ep)
    parts = [f"{name}={count}/{cov['steps']}"
             for name, count in sorted(cov["sidecars"].items())]
    keypoint_parts = [
        f"{name}={cov['keypoints_present'].get(name, 0)}/{total}"
        for name, total in sorted(cov["keypoints_total"].items())
    ]
    if keypoint_parts:
        parts.append("keypoints(" + " ".join(keypoint_parts) + ")")
    return "  ".join(parts) if parts else "annotations=none"


def replay_main(argv=None, prog=None) -> None:
    from lumen.data import CaseBundle, EpisodeDataset, replay, summarize

    parser = argparse.ArgumentParser(
        prog=prog, description="Summarize and replay a Lumen case-bundle corpus.")
    parser.add_argument("episodes_dir", nargs="?", default="episodes")
    args = parser.parse_args(argv)
    root = Path(args.episodes_dir)
    if not root.is_dir():
        print(f"no episodes under {str(root)!r}; run examples/capture_episode.py first")
        return
    ds = EpisodeDataset(root, validate_on_load=False)
    if len(ds) == 0:
        print(f"no episodes under {str(root)!r}; run examples/capture_episode.py first")
        return
    bundles = []
    skipped = []
    for d in ds.dirs:
        try:
            bundles.append(CaseBundle.load(d))
        except KeyError as e:
            skipped.append((d, f"manifest missing required key {e!s}"))
        except Exception as e:
            skipped.append((d, f"{type(e).__name__}: {e}"))
    if not bundles:
        print(f"no valid case bundles under {str(root)!r}")
        for path, err in skipped:
            print(f"  skipped {path}: {err}")
        return
    print(f"corpus: {summarize([b.episode for b in bundles])}\n")
    for bundle in bundles:
        ep = bundle.episode
        first_obs = next((obs for *_, obs in replay(ep) if obs is not None), None)
        shape = None if first_obs is None else first_obs.shape
        print(f"{ep.outcome.label:18s}  steps={ep.outcome.steps:2d}  "
              f"success={ep.outcome.success!s:5s}  final_dist={ep.outcome.final_dist:6.2f}  "
              f"obs{shape}  calib={bundle.calibration.get('type')}  "
              f"{_clinical_flags(ep)}  {_annotation_flags(ep)}  @ {ep.root}")
    if skipped:
        print("\nskipped invalid bundles:")
        for path, err in skipped:
            print(f"  {path}: {err}")


def validate_main(argv=None, prog=None) -> None:
    from lumen.data import Episode, EpisodeDataset, validate_case_bundle

    parser = argparse.ArgumentParser(
        prog=prog, description="Validate a Lumen case-bundle corpus before training.")
    parser.add_argument("episodes_dir", nargs="?", default="episodes")
    parser.add_argument("--require-cv-labels", action="store_true",
                        help="Require every fluoro observation to have device/vessel masks "
                             "and present tip/base keypoints.")
    args = parser.parse_args(argv)

    root = Path(args.episodes_dir)
    if not root.is_dir():
        print(f"no episodes under {str(root)!r}; run examples/capture_episode.py first")
        raise SystemExit(1)

    ds = EpisodeDataset(root, validate_on_load=False)
    valid = 0
    cv_steps = 0
    skipped = []
    for d in ds.dirs:
        try:
            ep = Episode.load(d)
            validate_case_bundle(ep, root=d)
            if args.require_cv_labels:
                cv_steps += _require_cv_labels(ep, d)
        except Exception as e:
            skipped.append((d, f"{type(e).__name__}: {e}"))
            continue
        valid += 1

    cv_msg = f"  cv_label_steps={cv_steps}" if args.require_cv_labels else ""
    print(f"validated {valid} case bundles under {root}{cv_msg}")
    if args.require_cv_labels and cv_steps == 0:
        skipped.append((root, "ValueError: no fluoro observations found for --require-cv-labels"))
    if skipped:
        print("invalid bundles:")
        for path, err in skipped:
            print(f"  {path}: {err}")
        raise SystemExit(1)
    if valid == 0:
        raise SystemExit(1)


def _require_cv_labels(ep, root) -> int:
    fluoro_steps = 0
    for i, step in enumerate(ep.steps):
        if step.obs_modality != "fluoro" or not step.obs_ref:
            continue
        fluoro_steps += 1
        annotations = step.annotations if isinstance(step.annotations, dict) else {}
        missing = [name for name in ("device_mask_ref", "vessel_mask_ref")
                   if not annotations.get(name)]
        for name in ("device_mask", "vessel_mask"):
            if annotations.get(f"{name}_ref"):
                mask = step.load_annotation(root, name)
                if mask is None or not mask.any():
                    missing.append(f"{name} nonempty")
        keypoints = annotations.get("keypoints") if isinstance(annotations.get("keypoints"), dict) else {}
        for name in ("tip", "base"):
            kp = keypoints.get(name)
            if not isinstance(kp, dict) or not kp.get("present", True):
                missing.append(f"keypoints.{name}")
        if missing:
            raise ValueError(f"step {i}: missing CV labels: {', '.join(missing)}")
    return fluoro_steps


def index_main(argv=None, prog=None) -> None:
    from lumen.data import Episode, EpisodeDataset, iter_step_records, validate_case_bundle

    parser = argparse.ArgumentParser(
        prog=prog, description="Write a JSONL index for a Lumen case-bundle corpus.")
    parser.add_argument("episodes_dir", nargs="?", default="episodes")
    parser.add_argument("--out", help="Output JSONL file. Defaults to stdout.")
    parser.add_argument("--absolute-paths", action="store_true",
                        help="Emit machine-local absolute sidecar paths instead of corpus-relative paths.")
    parser.add_argument("--check-sidecars", action="store_true",
                        help="Validate referenced arrays exist before indexing.")
    parser.add_argument("--modality", choices=("all", "fluoro", "luminal", "none"),
                        default="all",
                        help="Only emit rows for one observation modality. Defaults to all.")
    parser.add_argument("--require-cv-labels", action="store_true",
                        help="Require fluoro observations to have non-empty masks and "
                             "present tip/base keypoints before indexing.")
    args = parser.parse_args(argv)
    if args.require_cv_labels and args.modality not in ("all", "fluoro"):
        parser.error("--require-cv-labels is only valid with --modality all or fluoro")

    root = Path(args.episodes_dir)
    if not root.is_dir():
        print(f"no episodes under {str(root)!r}; run examples/capture_episode.py first")
        raise SystemExit(1)

    ds = EpisodeDataset(root, validate_on_load=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out, "w") if args.out else sys.stdout
    index_base_dir = None if args.absolute_paths else (Path(args.out).parent if args.out else root)
    records = episodes = contributing_episodes = 0
    cv_steps = 0
    skipped = []
    try:
        for d in ds.dirs:
            try:
                ep = Episode.load(d)
                check_root = d if (args.check_sidecars or args.require_cv_labels) else None
                validate_case_bundle(ep, root=check_root)
                if args.require_cv_labels:
                    cv_steps += _require_cv_labels(ep, d)
            except Exception as e:
                skipped.append((d, f"{type(e).__name__}: {e}"))
                continue
            episodes += 1
            episode_records = 0
            for record in iter_step_records(ep, d, base_dir=index_base_dir):
                if args.modality != "all" and record.get("obs_modality") != args.modality:
                    continue
                out.write(json.dumps(record, sort_keys=True) + "\n")
                records += 1
                episode_records += 1
            if episode_records:
                contributing_episodes += 1
    finally:
        if args.out:
            out.close()

    target = args.out or "stdout"
    cv_msg = f"  cv_label_steps={cv_steps}" if args.require_cv_labels else ""
    modality_msg = "" if args.modality == "all" else f"  modality={args.modality}"
    source = (f"{episodes} case bundles" if args.modality == "all"
              else f"{contributing_episodes}/{episodes} valid case bundles")
    msg = (f"indexed {records} step records from {source} -> "
           f"{target}{modality_msg}{cv_msg}")
    print(msg, file=(sys.stdout if args.out else sys.stderr))
    if args.require_cv_labels and cv_steps == 0 and not skipped:
        skipped.append((root, "ValueError: no fluoro observations found for --require-cv-labels"))
    if skipped:
        print("skipped invalid bundles:", file=(sys.stdout if args.out else sys.stderr))
        for path, err in skipped:
            print(f"  {path}: {err}", file=(sys.stdout if args.out else sys.stderr))
    if records == 0:
        print("no index records emitted; check the corpus path or modality filter",
              file=(sys.stdout if args.out else sys.stderr))
        if args.out:
            Path(args.out).unlink(missing_ok=True)
        raise SystemExit(1)
    if skipped:
        if args.check_sidecars or args.require_cv_labels:
            raise SystemExit(1)
    if (args.check_sidecars or args.require_cv_labels) and episodes == 0:
        raise SystemExit(1)


def calibrate_main(argv=None, prog=None) -> None:
    from lumen.data import EpisodeDataset, calibrate_from_episode, probe_episode
    from lumen.sensors import FluoroSensor
    from lumen.sensors.device_as_sensor import device_on_wall

    parser = argparse.ArgumentParser(
        prog=prog, description="Run the wall-probe calibration identifiability demo.")
    parser.parse_args(argv)
    true_C10 = 6.0e3
    sensor = FluoroSensor(mu_device=1.0, res=36, n_samples=90, nu=44, nv=44)
    nodes = device_on_wall(true_C10)
    cx = sensor.default_carm(nodes, axis=(1, 0, 0))
    cy = sensor.default_carm(nodes, axis=(0, 1, 0))

    import tempfile
    print("noise-free recovery is trivial (loss(true)=0); the honest test is under noise:\n")
    for name, carms in (("mono", [cx]), ("biplanar", [cx, cy])):
        with tempfile.TemporaryDirectory() as d:
            probe_episode(true_C10, sensor, carms=carms, notes={"case": name}).save(d)
            ep = EpisodeDataset(d)[0]
            res = calibrate_from_episode(ep, init_C10=2.0e3, iters=20, noise_std=1e-3)
            flag = "identifiable" if res["identifiable"] else "UNDER-DETERMINED"
            print(f"{name:9s}  views={res['n_views']}  recovered={res['recovered_C10']:.0f}  "
                  f"noise-free={res['rel_error']:.2%}  under-noise={res['rel_error_noisy']:.2%}  "
                  f"-> {flag}")
