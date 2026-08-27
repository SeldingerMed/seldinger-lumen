"""Frozen, license-clean procedural anatomy cases for the open repository."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable

from lumen.assets.schema import Asset


ANATOMY_PACK_VERSION = "lumen-anatomy/1"
ANATOMY_PACK_LICENSE = "Apache-2.0"
ANATOMY_PACK_PROVENANCE = "procedural"
ANATOMY_PACK_SOURCE = "lumen.assets.procedural"


@dataclass(frozen=True)
class AnatomyCase:
    """A named, reproducible procedural case and its license metadata."""

    case_id: str
    family: str
    description: str
    parameters: dict
    factory: Callable[[], Asset]

    def __post_init__(self):
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


    def manifest(self) -> dict:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "description": self.description,
            "parameters": dict(self.parameters),
            "provenance": ANATOMY_PACK_PROVENANCE,
            "license": ANATOMY_PACK_LICENSE,
            "source": ANATOMY_PACK_SOURCE,
        }

    def make(self) -> Asset:
        asset = self.factory()
        if not isinstance(asset, Asset):
            raise TypeError(f"anatomy case {self.case_id!r} did not return an Asset")
        if asset.provenance != ANATOMY_PACK_PROVENANCE:
            raise ValueError(f"anatomy case {self.case_id!r} is not procedural")
        return asset


def _procedural():
    from lumen.assets import procedural

    return procedural


ANATOMY_PACK = (
    AnatomyCase(
        "tube_nominal",
        "single_tube",
        "Straight constant-radius tube.",
        {"length": 100.0, "radius": 2.0, "n": 64},
        lambda: _procedural().straight_tube(length=100.0, radius=2.0, n=64),
    ),
    AnatomyCase(
        "tube_stenotic",
        "single_tube",
        "Straight tube with a moderate focal narrowing.",
        {"length": 100.0, "radius": 2.0, "severity": 0.6, "n": 96},
        lambda: _procedural().stenotic_tube(
            length=100.0, radius=2.0, severity=0.6, n=96
        ),
    ),
    AnatomyCase(
        "tube_tortuous",
        "single_tube",
        "Curved tapered tube with dilation and narrowing.",
        {
            "length": 100.0,
            "radius": 2.4,
            "severity": 0.35,
            "dilation": 0.16,
            "n": 96,
        },
        lambda: _procedural().tortuous_tube(
            length=100.0, radius=2.4, severity=0.35, dilation=0.16, n=96
        ),
    ),
    AnatomyCase(
        "tree_bifurcation",
        "branching_tree",
        "Symmetric Y bifurcation with two outlet branches.",
        {"trunk": 50.0, "branch": 50.0, "radius": 2.0, "angle_deg": 35.0, "n": 48},
        lambda: _procedural().bifurcation(
            trunk=50.0, branch=50.0, radius=2.0, angle_deg=35.0, n=48
        ),
    ),
    AnatomyCase(
        "tree_tortuous",
        "branching_tree",
        "Asymmetric curved tree with a side branch and focal narrowing.",
        {
            "radius": 4.0,
            "stenosis_severity": 0.30,
            "side_dilation": 0.28,
            "n": 44,
        },
        lambda: _procedural().tortuous_tree(
            radius=4.0, stenosis_severity=0.30, side_dilation=0.28, n=44
        ),
    ),
    AnatomyCase(
        "tree_aortic_arch",
        "branching_tree",
        "Procedural arch with descending and supra-arch branches.",
        {"radius": 5.0, "n": 48},
        lambda: _procedural().aortic_arch_tree(radius=5.0, n=48),
    ),
)


def anatomy_pack_manifest() -> dict:
    """Return the frozen pack manifest without materializing geometry."""
    return {
        "version": ANATOMY_PACK_VERSION,
        "license": ANATOMY_PACK_LICENSE,
        "provenance": ANATOMY_PACK_PROVENANCE,
        "source": ANATOMY_PACK_SOURCE,
        "cases": [case.manifest() for case in ANATOMY_PACK],
    }


def get_anatomy(case_id: str) -> Asset:
    """Materialize one named procedural case."""
    for case in ANATOMY_PACK:
        if case.case_id == case_id:
            return case.make()
    raise KeyError(f"unknown anatomy case {case_id!r}")


def materialize_anatomy_pack() -> dict[str, Asset]:
    """Materialize every case, retaining case IDs for reproducible callers."""
    return {case.case_id: case.make() for case in ANATOMY_PACK}


def validate_anatomy_pack() -> dict:
    """Validate pack metadata and materialize every case at the release boundary."""
    manifest = anatomy_pack_manifest()
    if manifest["version"] != ANATOMY_PACK_VERSION:
        raise ValueError("anatomy pack version mismatch")
    if manifest["license"] != ANATOMY_PACK_LICENSE:
        raise ValueError("anatomy pack license mismatch")
    case_ids = [case["case_id"] for case in manifest["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("anatomy pack case IDs must be unique")
    for case in ANATOMY_PACK:
        case.make()
    return manifest
