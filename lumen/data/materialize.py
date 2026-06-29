"""Materialize JSONL index rows into a portable NumPy training smoke-test batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from lumen.data.index import iter_index_records

DEFAULT_ARRAY_FIELDS = ("obs", "device_mask", "vessel_mask")
KEYPOINT_FIELDS = ("tip", "base")


def _as_uniform_stack(values: list[np.ndarray], field: str) -> np.ndarray:
    if not values:
        raise ValueError(f"no arrays collected for required field {field!r}")
    shape = values[0].shape
    dtype = values[0].dtype
    bad = [
        {"index": i, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        for i, arr in enumerate(values)
        if arr.shape != shape or arr.dtype != dtype
    ]
    if bad:
        expected = {"shape": list(shape), "dtype": str(dtype)}
        raise ValueError(
            f"field {field!r} is not uniform; expected {expected}, first mismatch {bad[0]}"
        )
    return np.stack(values, axis=0)


def _keypoint_uv(record: dict, name: str) -> list[float] | None:
    raw_keypoints = record.get("keypoints")
    keypoints = raw_keypoints if isinstance(raw_keypoints, dict) else {}
    kp = keypoints.get(name)
    if not isinstance(kp, dict) or not kp.get("present", True):
        return None
    uv = kp.get("uv")
    if uv is None:
        return None
    arr = np.asarray(uv, dtype=float)
    if arr.shape != (2,) or not np.isfinite(arr).all():
        return None
    return [float(arr[0]), float(arr[1])]


def _numeric_actions(record: dict) -> dict[str, float]:
    raw_action = record.get("action")
    action = raw_action if isinstance(raw_action, dict) else {}
    out = {}
    for key, value in action.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(key)] = float(value)
    return out


def _labels(record: dict) -> dict:
    labels = record.get("labels") if isinstance(record.get("labels"), dict) else {}
    return dict(labels) if labels else {"outcome": record.get("label")}


def materialize_index_batch(
    index_path: str | Path,
    out_npz: str | Path,
    *,
    limit: int | None = 32,
    fields: Iterable[str] = DEFAULT_ARRAY_FIELDS,
    base_dir: str | Path | None = None,
) -> dict:
    """Write a compact ``.npz`` batch from a Lumen JSONL index.

    The export is intentionally strict: every requested array field must be present
    on every selected row and must have the same shape and dtype. That makes the
    artifact safe to hand to a CV/RL training smoke test without discovering mixed
    tensor payloads after a dataloader has already started.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    field_names = tuple(dict.fromkeys(str(field) for field in fields))
    if not field_names:
        raise ValueError("at least one array field is required")

    arrays: dict[str, list[np.ndarray]] = {name: [] for name in field_names}
    metadata_rows = []
    action_keys: list[str] = []
    action_rows: list[dict[str, float]] = []
    keypoint_rows = {name: [] for name in KEYPOINT_FIELDS}

    for record in iter_index_records(index_path, load_arrays=True, base_dir=base_dir):
        missing = [name for name in field_names if name not in record]
        if missing:
            episode = record.get("episode", "<unknown>")
            step = record.get("step_index", "<unknown>")
            raise ValueError(
                f"index row episode {episode!r} step {step} is missing required arrays: "
                f"{', '.join(missing)}"
            )
        for name in field_names:
            arrays[name].append(np.asarray(record[name]))
        actions = _numeric_actions(record)
        for key in actions:
            if key not in action_keys:
                action_keys.append(key)
        action_rows.append(actions)
        for name in KEYPOINT_FIELDS:
            keypoint_rows[name].append(_keypoint_uv(record, name))
        metadata_rows.append({
            "episode": record.get("episode"),
            "step_index": record.get("step_index"),
            "t": record.get("t"),
            "label": record.get("label"),
            "obs_modality": record.get("obs_modality"),
            "labels": _labels(record),
            "outcome": record.get("outcome", {}),
        })
        if limit is not None and len(metadata_rows) >= limit:
            break

    if not metadata_rows:
        raise ValueError("no index records selected")

    payload = {name: _as_uniform_stack(values, name) for name, values in arrays.items()}
    action_matrix = np.full((len(metadata_rows), len(action_keys)), np.nan, dtype=float)
    for row_idx, actions in enumerate(action_rows):
        for col_idx, key in enumerate(action_keys):
            if key in actions:
                action_matrix[row_idx, col_idx] = actions[key]
    if action_keys:
        payload["actions"] = action_matrix
    for name, rows in keypoint_rows.items():
        if all(value is not None for value in rows):
            payload[f"{name}_uv"] = np.asarray(rows, dtype=float)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)  # type: ignore[arg-type]

    manifest = {
        "index_path": str(index_path),
        "out_npz": str(out_npz),
        "records": len(metadata_rows),
        "array_fields": list(field_names),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in payload.items()
        },
        "action_keys": action_keys,
        "rows": metadata_rows,
    }
    manifest_path = out_npz.with_suffix(out_npz.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
