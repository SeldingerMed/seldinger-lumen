"""Dataset card generation for Lumen dataloader indexes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lumen.data.index import summarize_index


def _format_counts(counts: dict) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _format_numeric(summary: dict, unit: str = "") -> str:
    if summary.get("mean") is None:
        return "-"
    return (f"mean={summary['mean']:.3f}{unit}, min={summary['min']:.3f}{unit}, "
            f"max={summary['max']:.3f}{unit}, n={summary['count']}")


def _format_percent(summary: dict) -> str:
    if summary.get("mean") is None:
        return "-"
    return (f"mean={summary['mean']:.3%}, min={summary['min']:.3%}, "
            f"max={summary['max']:.3%}, n={summary['count']}")


def _quality_findings(summary: dict) -> list[str]:
    findings = []
    if summary["records"] == 0:
        findings.append("index contains no records")
    for field, missing in summary.get("missing_paths", {}).items():
        if missing:
            findings.append(f"{field} has {missing} missing sidecar references")
    clinical = summary.get("clinical", {})
    if clinical.get("episode_inconsistencies"):
        findings.append("episode-level clinical endpoints are inconsistent across rows")
    annotations = summary.get("annotations", {})
    if annotations.get("cv_label_errors"):
        findings.append("fluoro rows are missing required CV labels")
    if annotations.get("keypoint_errors"):
        findings.append("keypoint QA found invalid or off-device keypoints")
    if summary.get("array_errors"):
        findings.append("array QA found malformed observations or masks")
    if summary.get("array_payload_errors"):
        findings.append("array payloads are not uniform for fixed-shape batching")
    return findings


def build_dataset_card(index_path: str | Path, *, title: str = "Lumen Dataset Card",
                       base_dir: str | Path | None = None, check_paths: bool = False,
                       check_arrays: bool = False, require_cv_labels: bool = False,
                       require_uniform_arrays: bool = False,
                       keypoint_mask_tolerance_px: float = 1.5) -> dict:
    """Return a Markdown dataset card plus the machine-readable summary used to make it."""
    summary = summarize_index(
        index_path,
        base_dir=base_dir,
        check_paths=check_paths,
        check_arrays=check_arrays,
        require_cv_labels=require_cv_labels,
        require_uniform_arrays=require_uniform_arrays,
        keypoint_mask_tolerance_px=keypoint_mask_tolerance_px,
    )
    findings = _quality_findings(summary)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    clinical = summary.get("clinical", {})
    annotations = summary.get("annotations", {})

    lines = [
        f"# {title}",
        "",
        f"Generated: {generated_at}",
        f"Index: `{summary['index_path']}`",
        "Provenance policy: this card is generated from index metadata; no PHI or private clinical data is embedded.",
        "",
        "## Corpus summary",
        "",
        f"- Records: {summary['records']}",
        f"- Episodes: {len(summary['episodes'])} ({_format_counts(summary['episodes'])})",
        f"- Modalities: {_format_counts(summary['modalities'])}",
        f"- Outcome labels: {_format_counts(summary['labels'])}",
        f"- Calibration types: {_format_counts(summary['calibration_types'])}",
        "",
        "## Clinical endpoints",
        "",
        f"- Outcome success: {_format_counts(clinical.get('outcome_success', {}))}",
        f"- Tip-target success: {_format_counts(clinical.get('tip_target_success', {}))}",
        f"- Wall perforation risk: {_format_counts(clinical.get('wall_perforation_risk', {}))}",
        f"- Final distance: {_format_numeric(clinical.get('final_dist', {}), 'mm')}",
        "",
        "## Annotation coverage",
        "",
        f"- Steps with keypoints: {annotations.get('keypoint_steps', 0)}/{summary['records']}",
        f"- Keypoints present: {_format_counts(annotations.get('keypoints_present', {}))}",
        f"- Keypoints total: {_format_counts(annotations.get('keypoints_total', {}))}",
        f"- CV labels required: {str(annotations.get('cv_labels_required', False)).lower()}",
        "",
        "## Sidecar and array QA",
        "",
        f"- Paths checked: {str(summary.get('paths_checked', False)).lower()}",
        f"- Arrays checked: {str(summary.get('arrays_checked', False)).lower()}",
    ]
    for field, count in summary.get("path_fields", {}).items():
        missing = summary.get("missing_paths", {}).get(field, 0)
        lines.append(f"- {field}: {count} refs, {missing} missing")
    if summary.get("array_payloads"):
        lines += ["", "Array payloads:"]
        for name, payloads in summary["array_payloads"].items():
            payload_text = ", ".join(
                f"{tuple(item['shape'])} {item['dtype']} n={item['count']}" for item in payloads
            )
            lines.append(f"- {name}: {payload_text}")
    if summary.get("mask_coverage"):
        lines += ["", "Mask coverage:"]
        for name, values in summary["mask_coverage"].items():
            lines.append(f"- {name}: {_format_percent(values)}")
    if summary.get("keypoint_device_distance"):
        lines += ["", "Device-keypoint distance:"]
        for name, values in summary["keypoint_device_distance"].items():
            lines.append(f"- {name}: {_format_numeric(values, 'px')}")
    lines += ["", "## Quality gate", ""]
    if findings:
        lines.append("Status: needs attention")
        lines.extend(f"- {finding}" for finding in findings)
    else:
        lines.append("Status: pass")
        lines.append("- No missing paths, label errors, endpoint inconsistencies, or array QA failures were found under the selected checks.")
    lines.append("")

    return {
        "title": title,
        "generated_at": generated_at,
        "summary": summary,
        "findings": findings,
        "markdown": "\n".join(lines),
    }


def write_dataset_card(card: dict, out_path: str | Path) -> str:
    """Write a generated dataset card to Markdown or JSON based on the output suffix."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        try:
            text = json.dumps(
                {k: v for k, v in card.items() if k != "markdown"},
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        except TypeError as e:
            raise ValueError(f"dataset card contains non-serializable data: {e}") from e
        out.write_text(text + "\n")
    else:
        out.write_text(card["markdown"])
    return str(out)
