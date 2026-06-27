"""Self-contained case bundles.

An Episode is the permissive interchange schema. A CaseBundle is the stricter
directory contract for CV/endo work: one folder must contain the anatomy asset,
sensor calibration, device definitions, observations, actions, outcome, and labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from lumen.data.replay import replay
from lumen.data.schema import Episode, _is_bare_file_ref, validate


def _modality(ep: Episode) -> str:
    if ep.meta.sensor.get("modality"):
        return ep.meta.sensor["modality"]
    for step in ep.steps:
        if step.obs_modality != "none":
            return step.obs_modality
    return "none"


def _validate_calibration(ep: Episode) -> None:
    cal = ep.meta.calibration
    modality = _modality(ep)
    if modality == "none":
        return
    if not cal:
        raise ValueError("case bundle missing calibration")
    typ = cal.get("type")
    if modality == "fluoro":
        if typ != "carm" or not cal.get("views"):
            raise ValueError("fluoro case bundle calibration must include C-arm views")
    elif modality == "luminal":
        if typ != "scope" or not cal.get("intrinsics"):
            raise ValueError("luminal case bundle calibration must include scope intrinsics")
    else:
        raise ValueError(f"case bundle has unsupported modality {modality!r}")


def validate_case_bundle(ep: Episode, root: str | None = None) -> None:
    """Validate the stricter self-contained bundle contract."""
    validate(ep, root=root)
    if not ep.meta.asset_ref:
        raise ValueError("case bundle missing asset_ref")
    if not _is_bare_file_ref(ep.meta.asset_ref):
        raise ValueError("case bundle asset_ref must be a local asset sidecar")
    if root is not None and ep.load_asset(root) is None:
        raise ValueError("case bundle asset_ref must point to a local asset sidecar")
    if not ep.meta.device:
        raise ValueError("case bundle missing device definitions")
    _validate_calibration(ep)
    if not (ep.outcome.label or ep.meta.labels):
        raise ValueError("case bundle missing labels")
    for i, step in enumerate(ep.steps):
        if not isinstance(step.action, dict):
            raise ValueError(f"step {i}: case bundle action missing")


@dataclass
class CaseBundle:
    """Loaded, replayable case directory."""
    root: str
    episode: Episode
    asset: object

    @classmethod
    def load(cls, root) -> "CaseBundle":
        root = str(root)
        ep = Episode.load(root)
        validate_case_bundle(ep, root=root)
        ep.root = root
        asset = ep.load_asset(root)
        if asset is None:
            raise ValueError("case bundle asset sidecar missing")
        return cls(root=root, episode=ep, asset=asset)

    @property
    def device_definitions(self) -> dict:
        return dict(self.episode.meta.device)

    @property
    def calibration(self) -> dict:
        return dict(self.episode.meta.calibration)

    @property
    def labels(self) -> dict:
        labels = {}
        if self.episode.outcome.label:
            labels["outcome"] = self.episode.outcome.label
        labels.update(self.episode.meta.labels)
        return labels

    def replay(self):
        return replay(self.episode, root=self.root)
