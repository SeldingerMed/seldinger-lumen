"""Installed command-line entry points for the common Lumen workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from lumen.data.index import KEYPOINT_MASK_TOLERANCE_PX, device_keypoint_mask_errors
from lumen.hardware import describe


def _command_table():
    return {
        "hardware": ("Print backend hardware/software status.", hardware_main),
        "benchmark": ("Run the canonical navigation benchmark.", benchmark_main),
        "play": ("Watch a scene: roll out a policy and write an animation.", play_main),
        "train": ("Train a navigation policy (CEM) and save it for play/eval.", train_main),
        "render-fluoro": ("Render the canonical synthetic fluoroscopy demo.", render_fluoro_main),
        "capture": ("Capture the canonical procedural case-bundle corpus.", capture_main),
        "replay": ("Summarize and replay a case-bundle corpus.", replay_main),
        "validate": ("Validate a case-bundle corpus before training.", validate_main),
        "index": ("Write a JSONL dataloader index for a corpus.", index_main),
        "inspect-index": ("Summarize and optionally path-check a JSONL dataloader index.",
                          inspect_index_main),
        "materialize-batch": ("Export a strict .npz smoke-test batch from a dataloader index.",
                              materialize_batch_main),
        "split-index": ("Write episode-grouped train/val/test splits for a JSONL index.",
                        split_index_main),
        "dataset-card": ("Generate a Markdown/JSON dataset card from a dataloader index.",
                         dataset_card_main),
        "calibrate": ("Run the wall-probe calibration identifiability demo.", calibrate_main),
        "import-mask": ("Import a segmented .npz volume as a Lumen asset.", import_mask_main),
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


def play_main(argv=None, prog=None) -> None:
    from lumen.viz import play

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Roll out a navigation scene under a policy and write a schematic "
                    "animation (<out>.avi + <out>.png). The one-command way to watch "
                    "the guidewire thread the vessel; reports the benchmark's "
                    "tip-reach and wall-safety numbers.")
    parser.add_argument("scene", nargs="?", default="tube",
                        choices=["tube", "stenotic", "tree"], help="which scene to play")
    parser.add_argument("--policy", default="forward",
                        help="forward | zero | random | <path>.npz (default: forward)")
    parser.add_argument("--steps", type=int, default=60, help="max steps to roll out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=480, help="frame size in pixels")
    parser.add_argument("--out", default="lumen_play", help="output path stem")
    args = parser.parse_args(argv)
    summary = play(scene=args.scene, policy=args.policy, steps=args.steps,
                   seed=args.seed, size=args.size, out=args.out)
    print(json.dumps(summary, indent=2))


def train_main(argv=None, prog=None) -> None:
    import numpy as np

    from lumen.assets import procedural
    from lumen.rl.cem import train_cem

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train a linear navigation policy with the gradient-free CEM over "
                    "the batched sim (no torch), and save it as an .npz you can hand to "
                    "`lumen play --policy`. CPU-friendly at the default sizes.")
    parser.add_argument("scene", nargs="?", default="tube",
                        choices=["tube", "stenotic"],
                        help="scene to train on (branch navigation is not a CEM target)")
    parser.add_argument("--pop", type=int, default=48, help="CEM population size")
    parser.add_argument("--iters", type=int, default=20, help="CEM iterations")
    parser.add_argument("--severity", type=float, default=0.5, help="stenosis severity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="lumen_policy.npz", help="output .npz path")
    args = parser.parse_args(argv)

    asset = (procedural.straight_tube(80.0, 2.0) if args.scene == "tube"
             else procedural.stenotic_tube(80.0, 2.0, severity=args.severity))
    pts, lumen = asset.edge_arrays(asset.edges[0])
    theta, hist = train_cem(
        np.asarray(pts), float(np.asarray(lumen.R).mean()), lumen_field=lumen,
        pop=args.pop, iters=args.iters, seed=args.seed, device="cpu",
        log=lambda r: print(f"  iter {r['iter']:2d}  return={r['mean_return']:+.3f}  "
                            f"success={r['success_rate']:.2f}", file=sys.stderr, flush=True))
    out = Path(args.out)
    if out.suffix != ".npz":
        out = out.with_suffix(".npz")
    np.savez(out, theta=np.asarray(theta, np.float32))
    final = hist[-1] if hist else {"success_rate": None}
    print(json.dumps({"scene": args.scene, "policy": str(out),
                      "iters": args.iters, "pop": args.pop,
                      "final_success_rate": final["success_rate"],
                      "play": f"lumen play {args.scene} --policy {out}"}, indent=2))


def render_fluoro_main(argv=None, prog=None) -> None:
    from lumen.workflows import render_fluoro_example

    parser = argparse.ArgumentParser(
        prog=prog, description="Render the canonical Lumen fluoro demo.")
    parser.add_argument("out_png", nargs="?", default="fluoro.png")
    args = parser.parse_args(argv)
    render_fluoro_example(args.out_png)


def import_mask_main(argv=None, prog=None) -> None:
    from lumen.assets import asset_from_mask, load_npz_volume, segment_threshold

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Convert a segmented .npz mask, or a thresholded .npz volume, "
                    "into a Lumen asset JSON. The .npz must contain 'mask' or "
                    "'volume', plus optional spacing_mm and origin_mm arrays.")
    parser.add_argument("input_npz")
    parser.add_argument("out_asset")
    parser.add_argument("--threshold", type=float,
                        help="Threshold a raw 'volume' array before import.")
    parser.add_argument("--foreground", choices=["above", "below"], default="above")
    parser.add_argument("--min-component-voxels", type=int, default=4)
    args = parser.parse_args(argv)

    vol = load_npz_volume(args.input_npz)
    mask_vol = (segment_threshold(vol, args.threshold, foreground=args.foreground)
                if args.threshold is not None else vol)
    asset = asset_from_mask(mask_vol.mask, spacing_mm=mask_vol.spacing_mm,
                            origin_mm=mask_vol.origin_mm,
                            min_component_voxels=args.min_component_voxels)
    asset.save(args.out_asset)
    print(f"wrote {args.out_asset}  nodes={len(asset.nodes)}  edges={len(asset.edges)}")


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


def _capture_hint(root: Path) -> str:
    return f"no episodes under {str(root)!r}; run `lumen capture {root}` first"


def replay_main(argv=None, prog=None) -> None:
    from lumen.data import CaseBundle, EpisodeDataset, replay, summarize

    parser = argparse.ArgumentParser(
        prog=prog, description="Summarize and replay a Lumen case-bundle corpus.")
    parser.add_argument("episodes_dir", nargs="?", default="episodes")
    args = parser.parse_args(argv)
    root = Path(args.episodes_dir)
    if not root.is_dir():
        print(_capture_hint(root))
        return
    ds = EpisodeDataset(root, validate_on_load=False)
    if len(ds) == 0:
        print(_capture_hint(root))
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
    parser.add_argument("--keypoint-mask-tolerance", type=float,
                        default=KEYPOINT_MASK_TOLERANCE_PX,
                        help="Max pixel distance from device keypoints to the device mask "
                             "when --require-cv-labels is enabled. Defaults to 1.5.")
    args = parser.parse_args(argv)
    if args.keypoint_mask_tolerance < 0:
        parser.error("--keypoint-mask-tolerance must be non-negative")

    root = Path(args.episodes_dir)
    if not root.is_dir():
        print(_capture_hint(root))
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
                cv_steps += _require_cv_labels(ep, d, args.keypoint_mask_tolerance)
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


def _require_cv_labels(ep, root, keypoint_mask_tolerance: float = KEYPOINT_MASK_TOLERANCE_PX) -> int:
    fluoro_steps = 0
    for i, step in enumerate(ep.steps):
        if step.obs_modality != "fluoro" or not step.obs_ref:
            continue
        fluoro_steps += 1
        annotations = step.annotations if isinstance(step.annotations, dict) else {}
        missing = [name for name in ("device_mask_ref", "vessel_mask_ref")
                   if not annotations.get(name)]
        device_mask = None
        for name in ("device_mask", "vessel_mask"):
            if annotations.get(f"{name}_ref"):
                mask = step.load_annotation(root, name)
                if mask is None or not mask.any():
                    missing.append(f"{name} nonempty")
                if name == "device_mask":
                    device_mask = mask
        keypoints = annotations.get("keypoints") if isinstance(annotations.get("keypoints"), dict) else {}
        for name in ("tip", "base"):
            kp = keypoints.get(name)
            if not isinstance(kp, dict) or not kp.get("present", True):
                missing.append(f"keypoints.{name}")
        missing.extend(device_keypoint_mask_errors(keypoints, device_mask, keypoint_mask_tolerance))
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
    parser.add_argument("--keypoint-mask-tolerance", type=float,
                        default=KEYPOINT_MASK_TOLERANCE_PX,
                        help="Max pixel distance from device keypoints to the device mask "
                             "when --require-cv-labels is enabled. Defaults to 1.5.")
    args = parser.parse_args(argv)
    if args.require_cv_labels and args.modality not in ("all", "fluoro"):
        parser.error("--require-cv-labels is only valid with --modality all or fluoro")
    if args.keypoint_mask_tolerance < 0:
        parser.error("--keypoint-mask-tolerance must be non-negative")

    root = Path(args.episodes_dir)
    if not root.is_dir():
        print(_capture_hint(root))
        raise SystemExit(1)

    ds = EpisodeDataset(root, validate_on_load=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    strict_checks = args.check_sidecars or args.require_cv_labels
    buffer_output = strict_checks
    out = None if buffer_output else (open(args.out, "w") if args.out else sys.stdout)
    output_lines = [] if buffer_output else None
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
                    cv_steps += _require_cv_labels(ep, d, args.keypoint_mask_tolerance)
            except Exception as e:
                skipped.append((d, f"{type(e).__name__}: {e}"))
                continue
            episodes += 1
            episode_records = 0
            for record in iter_step_records(ep, d, base_dir=index_base_dir):
                if args.modality != "all" and record.get("obs_modality") != args.modality:
                    continue
                line = json.dumps(record, sort_keys=True) + "\n"
                if output_lines is not None:
                    output_lines.append(line)
                else:
                    out.write(line)
                records += 1
                episode_records += 1
            if episode_records:
                contributing_episodes += 1
    finally:
        if args.out and out is not None:
            out.close()

    target = args.out or "stdout"
    cv_msg = f"  cv_label_steps={cv_steps}" if args.require_cv_labels else ""
    modality_msg = "" if args.modality == "all" else f"  modality={args.modality}"
    source = (f"{episodes} case bundles" if args.modality == "all"
              else f"{contributing_episodes}/{episodes} valid case bundles")
    if args.require_cv_labels and cv_steps == 0 and not skipped:
        skipped.append((root, "ValueError: no fluoro observations found for --require-cv-labels"))
    strict_failed = strict_checks and bool(skipped)
    msg_prefix = "index failed before writing" if strict_failed else "indexed"
    record_label = "candidate step records" if strict_failed else "step records"
    msg = (f"{msg_prefix} {records} {record_label} from {source} -> "
           f"{target}{modality_msg}{cv_msg}")

    def _print_skipped() -> None:
        print("skipped invalid bundles:", file=(sys.stdout if args.out else sys.stderr))
        for path, err in skipped:
            print(f"  {path}: {err}", file=(sys.stdout if args.out else sys.stderr))

    if records == 0:
        print(msg, file=(sys.stdout if args.out else sys.stderr))
        if skipped:
            _print_skipped()
        print("no index records emitted; check the corpus path or modality filter",
              file=(sys.stdout if args.out else sys.stderr))
        if args.out:
            Path(args.out).unlink(missing_ok=True)
        raise SystemExit(1)
    if strict_failed:
        print(msg, file=(sys.stdout if args.out else sys.stderr))
        _print_skipped()
        if args.out:
            Path(args.out).unlink(missing_ok=True)
        raise SystemExit(1)
    if (args.check_sidecars or args.require_cv_labels) and episodes == 0:
        raise SystemExit(1)
    if output_lines is not None:
        if args.out:
            with open(args.out, "w") as out_file:
                out_file.writelines(output_lines)
        else:
            sys.stdout.writelines(output_lines)
    print(msg, file=(sys.stdout if args.out else sys.stderr))
    if skipped:
        _print_skipped()


def split_index_main(argv=None, prog=None) -> None:
    from lumen.data import DEFAULT_RATIOS, DEFAULT_STRATIFY_FIELDS, split_index_records

    parser = argparse.ArgumentParser(
        prog=prog, description="Write train/val/test JSONL splits for a Lumen dataloader index.")
    parser.add_argument("index_path")
    parser.add_argument("--out-dir", default="splits",
                        help="Directory for train.jsonl, val.jsonl, test.jsonl, and manifest.json.")
    parser.add_argument("--ratios", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"),
                        default=DEFAULT_RATIOS,
                        help="Split ratios. Defaults to 0.8 0.1 0.1; values are normalized.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic shuffle seed for episode ordering.")
    parser.add_argument("--stratify", nargs="*", default=list(DEFAULT_STRATIFY_FIELDS),
                        help="Index record fields used for lightweight stratified ordering. "
                             "Defaults to label obs_modality.")
    parser.add_argument("--group-by", default="episode",
                        help="Record field kept together in one split. Defaults to episode.")
    args = parser.parse_args(argv)
    try:
        manifest = split_index_records(args.index_path, args.out_dir, ratios=args.ratios,
                                       seed=args.seed, stratify_fields=args.stratify,
                                       group_by=args.group_by)
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"could not split index {args.index_path!r}: {e}")
        raise SystemExit(1) from None

    print(f"split {manifest['records']} records from {manifest['episodes']} episodes -> {args.out_dir}")
    for split in ("train", "val", "test"):
        item = manifest["splits"][split]
        print(f"  {split}.jsonl: {item['records']} records, {item['episodes']} episodes")


def inspect_index_main(argv=None, prog=None) -> None:
    from lumen.data import summarize_index

    parser = argparse.ArgumentParser(
        prog=prog, description="Summarize a Lumen JSONL dataloader index.")
    parser.add_argument("index_path")
    parser.add_argument("--base-dir",
                        help="Resolve relative sidecar paths against this directory instead "
                             "of the index file's parent.")
    parser.add_argument("--check-paths", action="store_true",
                        help="Check that referenced observation/mask/node sidecars exist.")
    parser.add_argument("--check-arrays", action="store_true",
                        help="Load arrays and validate masks plus device-keypoint agreement.")
    parser.add_argument("--require-uniform-arrays", action="store_true",
                        help="Load arrays and fail if any array field has mixed shape/dtype payloads.")
    parser.add_argument("--keypoint-mask-tolerance", type=float,
                        default=KEYPOINT_MASK_TOLERANCE_PX,
                        help="Max pixel distance from device keypoints to the device mask "
                             "when --check-arrays is enabled. Defaults to 1.5.")
    parser.add_argument("--require-cv-labels", action="store_true",
                        help="Fail if fluoro rows lack mask refs or present tip/base keypoints.")
    parser.add_argument("--json", action="store_true",
                        help="Print the raw machine-readable summary JSON.")
    args = parser.parse_args(argv)
    if args.keypoint_mask_tolerance < 0:
        parser.error("--keypoint-mask-tolerance must be non-negative")

    try:
        summary = summarize_index(args.index_path, base_dir=args.base_dir,
                                  check_paths=args.check_paths,
                                  require_cv_labels=args.require_cv_labels,
                                  check_arrays=args.check_arrays,
                                  keypoint_mask_tolerance_px=args.keypoint_mask_tolerance,
                                  require_uniform_arrays=args.require_uniform_arrays)
    except FileNotFoundError:
        print(f"no index file at {args.index_path!r}")
        raise SystemExit(1) from None
    except ValueError as e:
        print(f"invalid index {args.index_path!r}: {e}")
        raise SystemExit(1) from None
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_index_summary(summary)
    if (summary["records"] == 0
            or any(summary["missing_paths"].values())
            or summary.get("clinical", {}).get("episode_inconsistencies")
            or summary.get("annotations", {}).get("cv_label_errors")
            or summary.get("annotations", {}).get("keypoint_errors")
            or summary.get("array_errors")
            or summary.get("array_payload_errors")):
        raise SystemExit(1)


def _format_counts(counts: dict) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _format_present_total(present: dict, total: dict) -> str:
    if not total:
        return "-"
    return ", ".join(f"{name}={present.get(name, 0)}/{count}"
                     for name, count in total.items())


def _format_fraction_summary(summary: dict) -> str:
    mean = summary.get("mean")
    if mean is None:
        return "-"
    return (f"mean={mean:.3%} min={summary['min']:.3%} "
            f"max={summary['max']:.3%} n={summary['count']}")


def _format_numeric_summary(summary: dict, unit: str = "") -> str:
    mean = summary.get("mean")
    if mean is None:
        return "-"
    return (f"mean={mean:.3f}{unit} min={summary['min']:.3f}{unit} "
            f"max={summary['max']:.3f}{unit} n={summary['count']}")


def _format_array_payloads(payloads: list[dict]) -> str:
    return ", ".join(
        f"{tuple(item['shape'])} {item['dtype']} n={item['count']}"
        for item in payloads
    )


def _print_index_summary(summary: dict) -> None:
    print(f"index: {summary['index_path']}")
    print(f"records: {summary['records']}")
    print(f"episodes: {len(summary['episodes'])} ({_format_counts(summary['episodes'])})")
    print(f"modalities: {_format_counts(summary['modalities'])}")
    print(f"labels: {_format_counts(summary['labels'])}")
    print(f"calibration_types: {_format_counts(summary['calibration_types'])}")
    clinical = summary.get("clinical", {})
    print("clinical (episodes):")
    print(f"  outcome_success: {_format_counts(clinical.get('outcome_success', {}))}")
    print(f"  tip_target_success: {_format_counts(clinical.get('tip_target_success', {}))}")
    print(f"  wall_perforation_risk: {_format_counts(clinical.get('wall_perforation_risk', {}))}")
    final_dist = clinical.get("final_dist", {})
    mean_dist = final_dist.get("mean")
    if mean_dist is None:
        print("  final_dist: -")
    else:
        print(f"  final_dist: mean={mean_dist:.3f} min={final_dist['min']:.3f} "
              f"max={final_dist['max']:.3f} n={final_dist['count']}")
    if clinical.get("episode_inconsistencies"):
        print("  endpoint inconsistencies:")
        for item in clinical["episode_inconsistencies"]:
            print(f"    {item['episode']}: line {item['line']} differs from line {item['first_line']}")
    annotations = summary.get("annotations", {})
    print("annotations:")
    print(f"  keypoint_steps: {annotations.get('keypoint_steps', 0)}/{summary['records']}")
    print("  keypoints: " + _format_present_total(
        annotations.get("keypoints_present", {}),
        annotations.get("keypoints_total", {}),
    ))
    if annotations.get("cv_labels_required"):
        print("  cv_labels_required: true")
    if annotations.get("cv_label_errors"):
        print("  cv label errors:")
        for item in annotations["cv_label_errors"]:
            print(f"    line {item['line']} {item.get('episode')}: "
                  f"missing {', '.join(item['missing'])}")
    if annotations.get("keypoint_errors"):
        print("  keypoint errors:")
        for item in annotations["keypoint_errors"]:
            print(f"    line {item['line']} {item.get('episode')}: "
                  f"{'; '.join(item['errors'])}")
    path_status = "checked" if summary["paths_checked"] or summary.get("arrays_checked") else "not checked"
    print(f"paths: {path_status}")
    for field, count in summary["path_fields"].items():
        missing = summary["missing_paths"].get(field, 0)
        print(f"  {field}: {count} refs, {missing} missing")
    if summary["missing_path_examples"]:
        print("missing examples:")
        for item in summary["missing_path_examples"]:
            print(f"  line {item['line']} {item.get('episode')}: "
                  f"{item['field']} -> {item['path']}")
    if summary.get("arrays_checked"):
        print("arrays: checked")
        payloads = summary.get("array_payloads", {})
        if payloads:
            print("array payloads:")
            for name, values in payloads.items():
                print(f"  {name}: {_format_array_payloads(values)}")
        if summary.get("arrays_uniform_required"):
            print("array_uniform_required: true")
        if summary.get("array_payload_errors"):
            print("array payload errors:")
            for item in summary["array_payload_errors"]:
                print(f"  {item['name']}: {_format_array_payloads(item['payloads'])}")
        print(f"keypoint_mask_tolerance: "
              f"{annotations.get('keypoint_mask_tolerance_px', 1.5):.3f}px")
        coverage = summary.get("mask_coverage", {})
        if coverage:
            print("mask coverage:")
            for name, values in coverage.items():
                print(f"  {name}: {_format_fraction_summary(values)}")
        distances = summary.get("keypoint_device_distance", {})
        if distances:
            print("keypoint device distance:")
            for name, values in distances.items():
                print(f"  {name}: {_format_numeric_summary(values, 'px')}")
    if summary.get("array_errors"):
        print("array errors:")
        for item in summary["array_errors"]:
            print(f"  line {item['line']} {item.get('episode')}: "
                  f"{'; '.join(item['errors'])}")


def materialize_batch_main(argv=None, prog=None) -> None:
    from lumen.data import materialize_index_batch

    parser = argparse.ArgumentParser(
        prog=prog, description="Export a strict .npz training smoke-test batch from a Lumen index.")
    parser.add_argument("index_path", help="JSONL index produced by `lumen index`.")
    parser.add_argument("out_npz", help="Output compressed .npz path.")
    parser.add_argument("--limit", type=int, default=32,
                        help="Maximum rows to export. Defaults to 32.")
    parser.add_argument("--fields", default="obs,device_mask,vessel_mask",
                        help="Comma-separated required array fields. Defaults to obs,device_mask,vessel_mask.")
    parser.add_argument("--base-dir",
                        help="Resolve index-relative paths against this directory instead of the index parent.")
    args = parser.parse_args(argv)
    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    try:
        manifest = materialize_index_batch(
            args.index_path,
            args.out_npz,
            limit=args.limit,
            fields=fields,
            base_dir=args.base_dir,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"could not materialize batch: {e}")
        raise SystemExit(1) from None
    arrays = ", ".join(
        f"{name}{tuple(info['shape'])} {info['dtype']}"
        for name, info in manifest["arrays"].items()
    )
    print(f"materialized {manifest['records']} records -> {manifest['out_npz']}")
    print(f"manifest: {manifest['manifest_path']}")
    print(f"arrays: {arrays}")


def dataset_card_main(argv=None, prog=None) -> None:
    from lumen.data import build_dataset_card, write_dataset_card

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Generate a shareable Markdown or JSON dataset card from a Lumen index.")
    parser.add_argument("index_path", help="JSONL index produced by `lumen index`.")
    parser.add_argument("--out", default="DATASET_CARD.md",
                        help="Output .md or .json path. Defaults to DATASET_CARD.md.")
    parser.add_argument("--title", default="Lumen Dataset Card")
    parser.add_argument("--base-dir",
                        help="Resolve index-relative paths against this directory instead of the index parent.")
    parser.add_argument("--check-paths", action="store_true",
                        help="Check referenced observation/mask/node sidecars exist before writing the card.")
    parser.add_argument("--check-arrays", action="store_true",
                        help="Load arrays and include payload, mask coverage, and keypoint-distance QA.")
    parser.add_argument("--require-cv-labels", action="store_true",
                        help="Mark the card failed if fluoro rows lack masks or tip/base keypoints.")
    parser.add_argument("--require-uniform-arrays", action="store_true",
                        help="Load arrays and mark the card failed if payload shape/dtype mixes.")
    parser.add_argument("--keypoint-mask-tolerance", type=float,
                        default=KEYPOINT_MASK_TOLERANCE_PX,
                        help="Max pixel distance from device keypoints to the device mask. Defaults to 1.5.")
    args = parser.parse_args(argv)
    if args.keypoint_mask_tolerance < 0:
        parser.error("--keypoint-mask-tolerance must be non-negative")
    try:
        card = build_dataset_card(
            args.index_path,
            title=args.title,
            base_dir=args.base_dir,
            check_paths=args.check_paths,
            check_arrays=args.check_arrays,
            require_cv_labels=args.require_cv_labels,
            require_uniform_arrays=args.require_uniform_arrays,
            keypoint_mask_tolerance_px=args.keypoint_mask_tolerance,
        )
    except FileNotFoundError:
        print(f"no index file at {args.index_path!r}")
        raise SystemExit(1) from None
    except ValueError as e:
        print(f"invalid index {args.index_path!r}: {e}")
        raise SystemExit(1) from None
    try:
        out = write_dataset_card(card, args.out)
    except Exception as e:
        print(f"could not write dataset card: {type(e).__name__}: {e}")
        raise SystemExit(1) from None
    status = "pass" if not card["findings"] else "needs attention"
    print(f"wrote dataset card: {out}")
    print(f"quality_gate: {status}")
    if card["findings"]:
        for finding in card["findings"]:
            print(f"  - {finding}")


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


if __name__ == "__main__":
    main()
