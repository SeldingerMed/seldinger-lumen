"""Machine-readable corpus indexes for CV/RL dataloaders."""

from __future__ import annotations

from pathlib import Path

from lumen.data.schema import Episode, _safe_path


def _sidecar_path(root: str | Path, ref: str | None) -> str | None:
    if not ref:
        return None
    return str(Path(_safe_path(str(root), ref)).resolve())


def _labels(ep: Episode) -> dict:
    labels = dict(ep.meta.labels)
    labels["outcome"] = ep.outcome.label
    return labels


def iter_step_records(ep: Episode, root: str | Path):
    """Yield JSON-serializable per-step records for one episode directory.

    The records are manifest-derived paths and metadata. They do not load image,
    mask, or node arrays, so callers can build an index cheaply and let their
    training dataloader decide when to open sidecars.
    """
    root = Path(root)
    episode_dir = str(root.resolve())
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
            "obs_path": _sidecar_path(root, step.obs_ref),
            "device_mask_path": _sidecar_path(root, annotations.get("device_mask_ref")),
            "node_positions_path": _sidecar_path(root, step.kinematics.get("node_positions_ref")),
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
