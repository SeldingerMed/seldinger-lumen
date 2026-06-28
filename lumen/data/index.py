"""Machine-readable corpus indexes for CV/RL dataloaders."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from lumen.data.schema import Episode, _safe_path


PATH_FIELDS = ("obs_path", "device_mask_path", "vessel_mask_path", "node_positions_path")


def _record_path(path: str | Path, base_dir: str | Path | None = None) -> str:
    path = Path(path).resolve()
    if base_dir is None:
        return str(path)
    return os.path.relpath(path, Path(base_dir).resolve())


def _sidecar_path(root: str | Path, ref: str | None, base_dir: str | Path | None = None) -> str | None:
    if not ref:
        return None
    return _record_path(_safe_path(str(root), ref), base_dir)


def _labels(ep: Episode) -> dict:
    labels = dict(ep.meta.labels)
    labels["outcome"] = ep.outcome.label
    return labels


def iter_step_records(ep: Episode, root: str | Path, base_dir: str | Path | None = None):
    """Yield JSON-serializable per-step records for one episode directory.

    The records are manifest-derived paths and metadata. They do not load image,
    mask, or node arrays, so callers can build an index cheaply and let their
    training dataloader decide when to open sidecars. Pass ``base_dir`` to emit
    paths relative to the index location or corpus root; omit it for absolute paths.
    """
    root = Path(root)
    episode_dir = _record_path(root, base_dir)
    labels = _labels(ep)
    outcome = {
        "success": ep.outcome.success,
        "final_dist": ep.outcome.final_dist,
        "steps": ep.outcome.steps,
        "retrieval": ep.outcome.retrieval,
        "label": ep.outcome.label,
    }
    calibration = ep.meta.calibration if isinstance(ep.meta.calibration, dict) else {}
    for i, step in enumerate(ep.steps):
        annotations = step.annotations if isinstance(step.annotations, dict) else {}
        yield {
            "episode": root.name,
            "episode_dir": episode_dir,
            "label": ep.outcome.label,
            "step_index": i,
            "t": step.t,
            "obs_modality": step.obs_modality,
            "obs_path": _sidecar_path(root, step.obs_ref, base_dir),
            "device_mask_path": _sidecar_path(root, annotations.get("device_mask_ref"), base_dir),
            "vessel_mask_path": _sidecar_path(root, annotations.get("vessel_mask_ref"), base_dir),
            "node_positions_path": _sidecar_path(root, step.kinematics.get("node_positions_ref"), base_dir),
            "keypoints": annotations.get("keypoints", {}),
            "action": dict(step.action),
            "kinematics": dict(step.kinematics),
            "labels": dict(labels),
            "outcome": dict(outcome),
            "clinical_metrics": dict(ep.outcome.metrics),
            "calibration_type": calibration.get("type"),
            "provenance": ep.meta.provenance,
            "version": ep.meta.version,
        }


def resolve_record_paths(record: dict, base_dir: str | Path) -> dict:
    """Return a copy with relative ``*_path`` fields resolved under ``base_dir``."""
    out = dict(record)
    base = Path(base_dir)
    for key, value in record.items():
        if not key.endswith("_path") or not value:
            continue
        path = Path(value)
        out[key] = str(path if path.is_absolute() else (base / path).resolve())
    return out


def load_step_record(record: dict, base_dir: str | Path | None = None) -> dict:
    """Load arrays referenced by one JSONL index row.

    Returns a copy of the record with resolved path fields and array entries:
    ``obs``, ``device_mask``, ``vessel_mask``, and ``node_positions`` when the
    corresponding path exists in the row.
    """
    sample = resolve_record_paths(record, base_dir or ".")
    for key, name in (
        ("obs_path", "obs"),
        ("device_mask_path", "device_mask"),
        ("vessel_mask_path", "vessel_mask"),
        ("node_positions_path", "node_positions"),
    ):
        path = sample.get(key)
        if path:
            sample[name] = np.load(path)
    return sample


def iter_index_records(index_path: str | Path, load_arrays: bool = False,
                       base_dir: str | Path | None = None):
    """Iterate a ``lumen-index`` JSONL file.

    Relative sidecar paths are resolved against ``base_dir``. If ``base_dir`` is
    omitted, they are resolved against the directory containing the index file,
    which matches the recommended ``episodes/index.jsonl`` layout.
    """
    index_path = Path(index_path)
    root = Path(base_dir) if base_dir is not None else index_path.parent
    with open(index_path) as f:
        for line in f:
            record = json.loads(line)
            yield load_step_record(record, root) if load_arrays else resolve_record_paths(record, root)


def _counter_dict(counter: Counter) -> dict:
    return {str(k): counter[k] for k in sorted(counter, key=str)}


def summarize_index(index_path: str | Path, base_dir: str | Path | None = None,
                    check_paths: bool = False) -> dict:
    """Return a compact JSON-serializable summary of a Lumen dataloader index."""
    index_path = Path(index_path)
    root = Path(base_dir) if base_dir is not None else index_path.parent
    episodes = Counter()
    modalities = Counter()
    labels = Counter()
    calibration_types = Counter()
    path_counts = Counter({field: 0 for field in PATH_FIELDS})
    missing_paths = Counter({field: 0 for field in PATH_FIELDS})
    missing_examples = []
    records = 0
    with open(index_path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"line {line_no}: invalid JSON: {e.msg}") from e
            if not isinstance(record, dict):
                raise ValueError(
                    f"line {line_no}: expected JSON object, got {type(record).__name__}")
            records += 1
            episodes[record.get("episode", "<missing>")] += 1
            modalities[record.get("obs_modality", "<missing>")] += 1
            labels[record.get("label", "<missing>")] += 1
            calibration_types[record.get("calibration_type", "<missing>")] += 1
            resolved = resolve_record_paths(record, root) if check_paths else record
            for field in PATH_FIELDS:
                value = record.get(field)
                if not value:
                    continue
                path_counts[field] += 1
                if check_paths and not Path(resolved[field]).exists():
                    missing_paths[field] += 1
                    if len(missing_examples) < 5:
                        missing_examples.append({
                            "line": line_no,
                            "episode": record.get("episode"),
                            "field": field,
                            "path": resolved[field],
                        })
    return {
        "index_path": str(index_path),
        "records": records,
        "episodes": _counter_dict(episodes),
        "modalities": _counter_dict(modalities),
        "labels": _counter_dict(labels),
        "calibration_types": _counter_dict(calibration_types),
        "path_fields": {field: path_counts[field] for field in PATH_FIELDS},
        "paths_checked": check_paths,
        "missing_paths": {field: missing_paths[field] for field in PATH_FIELDS},
        "missing_path_examples": missing_examples,
    }
