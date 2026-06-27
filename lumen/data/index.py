"""Machine-readable corpus indexes for CV/RL dataloaders."""

from __future__ import annotations

from pathlib import Path

from lumen.data.schema import Episode, _safe_path


def _record_path(path: str | Path, base_dir: str | Path | None = None) -> str:
    path = Path(path).resolve()
    if base_dir is None:
        return str(path)
    return str(path.relative_to(Path(base_dir).resolve()))


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
    paths relative to a corpus root; omit it for absolute paths.
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
            "step_index": i,
            "t": step.t,
            "obs_modality": step.obs_modality,
            "obs_path": _sidecar_path(root, step.obs_ref, base_dir),
            "device_mask_path": _sidecar_path(root, annotations.get("device_mask_ref"), base_dir),
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
