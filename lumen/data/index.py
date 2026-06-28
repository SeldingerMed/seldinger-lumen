"""Machine-readable corpus indexes for CV/RL dataloaders."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from lumen.data.schema import Episode, _safe_path


PATH_FIELDS = ("obs_path", "device_mask_path", "vessel_mask_path", "node_positions_path")
CV_KEYPOINTS = ("base", "tip")
DEVICE_KEYPOINTS = ("base", "tip", "nodes")
KEYPOINT_MASK_TOLERANCE_PX = 1.5


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
    episode = sample.get("episode", "<unknown>")
    step = sample.get("step_index", "<unknown>")
    for key, name in (
        ("obs_path", "obs"),
        ("device_mask_path", "device_mask"),
        ("vessel_mask_path", "vessel_mask"),
        ("node_positions_path", "node_positions"),
    ):
        path = sample.get(key)
        if path:
            try:
                sample[name] = np.load(path)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"missing {key} for episode {episode!r} step {step}: {path}") from e
            except Exception as e:
                raise ValueError(
                    f"could not load {key} for episode {episode!r} step {step}: "
                    f"{path} ({type(e).__name__}: {e})") from e
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


def _bool_key(value) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "missing"


def _numeric_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _count_keypoints(keypoints, present: Counter, total: Counter) -> bool:
    if not isinstance(keypoints, dict) or not keypoints:
        return False
    counted = False
    for name, value in keypoints.items():
        values = value if isinstance(value, list) else [value]
        for kp in values:
            if not isinstance(kp, dict):
                continue
            total[str(name)] += 1
            present[str(name)] += bool(kp.get("present", True))
            counted = True
    return counted


def _cv_label_errors(record: dict) -> list[str]:
    if record.get("obs_modality") != "fluoro":
        return []
    errors = []
    for field in ("device_mask_path", "vessel_mask_path"):
        if not record.get(field):
            errors.append(field)
    keypoints = record.get("keypoints") if isinstance(record.get("keypoints"), dict) else {}
    for name in CV_KEYPOINTS:
        kp = keypoints.get(name)
        if not isinstance(kp, dict) or not kp.get("present", True):
            errors.append(f"keypoints.{name}")
    return errors


def _nearest_mask_distance(mask: np.ndarray, uv: np.ndarray) -> float | None:
    if mask.ndim != 2 or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(np.sqrt(((xs - uv[0]) ** 2 + (ys - uv[1]) ** 2).min()))


def _keypoint_errors(record: dict, obs_shape: tuple | None = None,
                     device_mask: np.ndarray | None = None) -> list[str]:
    keypoints = record.get("keypoints")
    if keypoints in (None, {}):
        return []
    if not isinstance(keypoints, dict):
        return ["keypoints must be mapping"]
    errors = []
    for name, value in keypoints.items():
        values = value if isinstance(value, list) else [value]
        for j, kp in enumerate(values):
            label = f"keypoints.{name}[{j}]" if isinstance(value, list) else f"keypoints.{name}"
            if not isinstance(kp, dict):
                errors.append(f"{label} must be mapping")
                continue
            present = kp.get("present", True)
            if not isinstance(present, bool):
                errors.append(f"{label} present must be bool")
            uv = kp.get("uv")
            if uv is None:
                if present:
                    errors.append(f"{label} uv")
                continue
            try:
                arr = np.asarray(uv, dtype=float)
            except (TypeError, ValueError):
                errors.append(f"{label} uv numeric")
                continue
            if arr.shape != (2,):
                errors.append(f"{label} uv length")
                continue
            if not np.isfinite(arr).all():
                errors.append(f"{label} uv finite")
                continue
            if present and obs_shape is not None:
                h, w = obs_shape
                u, v = arr
                if not (0.0 <= u < w and 0.0 <= v < h):
                    errors.append(f"{label} in-frame")
                    continue
            if present and name in DEVICE_KEYPOINTS and device_mask is not None:
                dist = _nearest_mask_distance(np.asarray(device_mask) > 0, arr)
                if dist is not None and dist > KEYPOINT_MASK_TOLERANCE_PX:
                    errors.append(f"{label} on-device distance={dist:.2f}px")
    return errors


def _array_errors(record: dict, resolved: dict,
                  mask_coverage: dict | None = None) -> tuple[list[str], tuple | None, np.ndarray | None]:
    errors = []
    arrays = {}
    for field in PATH_FIELDS:
        value = record.get(field)
        if not value:
            continue
        if not Path(resolved[field]).exists():
            continue
        try:
            arrays[field] = np.load(resolved[field])
        except Exception as e:
            errors.append(f"{field} load: {type(e).__name__}: {e}")
    obs = arrays.get("obs_path")
    obs_shape = None if obs is None else tuple(obs.shape[:2])
    for field in ("device_mask_path", "vessel_mask_path"):
        if field not in arrays:
            continue
        mask = np.asarray(arrays[field])
        name = field.removesuffix("_path")
        if mask.ndim != 2:
            errors.append(f"{name} ndim={mask.ndim}")
        if mask.dtype.kind not in ("b", "u"):
            errors.append(f"{name} dtype={mask.dtype}")
        if obs_shape is not None and tuple(mask.shape) != obs_shape:
            errors.append(f"{name} shape={mask.shape} obs_shape={obs_shape}")
        if not mask.any():
            errors.append(f"{name} nonempty")
        if mask_coverage is not None and mask.ndim == 2 and mask.size:
            mask_coverage.setdefault(name, []).append(float(np.count_nonzero(mask) / mask.size))
    return errors, obs_shape, arrays.get("device_mask_path")


def summarize_index(index_path: str | Path, base_dir: str | Path | None = None,
                    check_paths: bool = False, require_cv_labels: bool = False,
                    check_arrays: bool = False) -> dict:
    """Return a compact JSON-serializable summary of a Lumen dataloader index."""
    index_path = Path(index_path)
    root = Path(base_dir) if base_dir is not None else index_path.parent
    episodes = Counter()
    modalities = Counter()
    labels = Counter()
    calibration_types = Counter()
    path_counts = Counter({field: 0 for field in PATH_FIELDS})
    missing_paths = Counter({field: 0 for field in PATH_FIELDS})
    outcome_success = Counter()
    tip_target_success = Counter()
    wall_perforation_risk = Counter()
    final_dists = []
    clinical_by_episode = {}
    clinical_inconsistencies = []
    keypoint_steps = 0
    keypoints_present = Counter()
    keypoints_total = Counter()
    cv_label_errors = []
    keypoint_errors = []
    array_errors = []
    mask_coverage = {}
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
            episode_name = record.get("episode", "<missing>")
            outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
            clinical = (record.get("clinical_metrics")
                        if isinstance(record.get("clinical_metrics"), dict) else {})
            tip_target = (clinical.get("tip_target")
                          if isinstance(clinical.get("tip_target"), dict) else {})
            wall_safety = (clinical.get("wall_safety")
                           if isinstance(clinical.get("wall_safety"), dict) else {})
            final_dist = outcome.get("final_dist")
            final_dist_value = (float(final_dist)
                                if isinstance(final_dist, (int, float))
                                and not isinstance(final_dist, bool)
                                else None)
            endpoint = {
                "outcome_success": _bool_key(outcome.get("success")),
                "tip_target_success": _bool_key(tip_target.get("success")),
                "wall_perforation_risk": _bool_key(wall_safety.get("perforation_risk")),
                "final_dist": final_dist_value,
            }
            if episode_name not in clinical_by_episode:
                clinical_by_episode[episode_name] = {"line": line_no, "endpoint": endpoint}
                outcome_success[endpoint["outcome_success"]] += 1
                tip_target_success[endpoint["tip_target_success"]] += 1
                wall_perforation_risk[endpoint["wall_perforation_risk"]] += 1
                if final_dist_value is not None:
                    final_dists.append(final_dist_value)
            elif clinical_by_episode[episode_name]["endpoint"] != endpoint and len(clinical_inconsistencies) < 5:
                clinical_inconsistencies.append({
                    "episode": episode_name,
                    "first_line": clinical_by_episode[episode_name]["line"],
                    "line": line_no,
                })
            if _count_keypoints(record.get("keypoints"), keypoints_present, keypoints_total):
                keypoint_steps += 1
            if require_cv_labels:
                errors = _cv_label_errors(record)
                if errors and len(cv_label_errors) < 5:
                    cv_label_errors.append({
                        "line": line_no,
                        "episode": record.get("episode"),
                        "missing": errors,
                    })
            resolved = resolve_record_paths(record, root) if (check_paths or check_arrays) else record
            for field in PATH_FIELDS:
                value = record.get(field)
                if not value:
                    continue
                path_counts[field] += 1
                if (check_paths or check_arrays) and not Path(resolved[field]).exists():
                    missing_paths[field] += 1
                    if len(missing_examples) < 5:
                        missing_examples.append({
                            "line": line_no,
                            "episode": record.get("episode"),
                            "field": field,
                            "path": resolved[field],
                        })
            obs_shape = None
            device_mask = None
            if check_arrays:
                errors, obs_shape, device_mask = _array_errors(record, resolved, mask_coverage)
                if errors and len(array_errors) < 5:
                    array_errors.append({
                        "line": line_no,
                        "episode": record.get("episode"),
                        "errors": errors,
                    })
            if require_cv_labels or check_arrays:
                errors = _keypoint_errors(record, obs_shape, device_mask)
                if errors and len(keypoint_errors) < 5:
                    keypoint_errors.append({
                        "line": line_no,
                        "episode": record.get("episode"),
                        "errors": errors,
                    })
    return {
        "index_path": str(index_path),
        "records": records,
        "episodes": _counter_dict(episodes),
        "modalities": _counter_dict(modalities),
        "labels": _counter_dict(labels),
        "calibration_types": _counter_dict(calibration_types),
        "path_fields": {field: path_counts[field] for field in PATH_FIELDS},
        "clinical": {
            "outcome_success": _counter_dict(outcome_success),
            "tip_target_success": _counter_dict(tip_target_success),
            "wall_perforation_risk": _counter_dict(wall_perforation_risk),
            "final_dist": _numeric_summary(final_dists),
            "episode_inconsistencies": clinical_inconsistencies,
        },
        "annotations": {
            "keypoint_steps": keypoint_steps,
            "keypoints_present": _counter_dict(keypoints_present),
            "keypoints_total": _counter_dict(keypoints_total),
            "cv_labels_required": require_cv_labels,
            "cv_label_errors": cv_label_errors,
            "keypoint_errors": keypoint_errors,
        },
        "paths_checked": check_paths or check_arrays,
        "arrays_checked": check_arrays,
        "array_errors": array_errors,
        "mask_coverage": (
            {name: _numeric_summary(values) for name, values in sorted(mask_coverage.items())}
            if check_arrays else {}
        ),
        "missing_paths": {field: missing_paths[field] for field in PATH_FIELDS},
        "missing_path_examples": missing_examples,
    }
