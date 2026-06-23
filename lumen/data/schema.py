"""The episode schema — Layer 2's data standard (doc §5).

There is no Open X-Embodiment for intervention. Layer 2's deliverable is the
*standard* for a captured case: a time-synchronized log of device kinematics, the
paired observation (the Layer-1 fluoro / luminal frame), and the outcome, over a
declared coordinate frame. Whoever defines the schema and accumulates the corpus
owns the calibration flywheel (§327).

Like the asset schema, BOTH ends emit this same object:
  * the open `lumen.data.capture` recorder (provenance = "procedural")
  * a private patient-capture pipeline in seldinger-ml (provenance = "patient(private)")
Patient-derived episodes MUST NOT live in this open repo; the firewall enforces
`provenance == "procedural"` on every committed manifest (it scans all *.json).

On-disk layout (one directory per episode):
    <root>/manifest.json     scalars: meta, per-step kinematics/actions/outcome, refs
    <root>/obs/<name>.npy     observation + node-position sidecars (lazy-loaded)
Observations are stored as .npy (lossless, dependency-free for both grayscale fluoro
and RGB luminal); a viewer PNG is an example-side extra, not part of the load path.

Carries geometry/observation/kinematics/outcome only — wall mechanics (HGO) are the
CALIBRATION TARGET and stay private (§8), exactly as in the asset schema.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from lumen.assets.schema import Frame

SCHEMA_VERSION = "lumen-episode/0"


@dataclass
class EpisodeMeta:
    frame: Frame = field(default_factory=Frame)   # declared coordinate convention (reused)
    asset_ref: str = ""                           # path/id of the lumen-asset/0 geometry
    device: dict = field(default_factory=dict)    # device knobs (radius, stiffness, ...)
    sensor: dict = field(default_factory=dict)    # modality + render params
    dt: float = 0.0                               # sim timestep per recorded step
    notes: dict = field(default_factory=dict)     # free-form (sim2sim ground truth lives here)
    provenance: str = "procedural"                # "procedural" | "patient(private)"
    version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeMeta":
        d = dict(d)
        d["frame"] = Frame(**d["frame"]) if isinstance(d.get("frame"), dict) else Frame()
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Step:
    """One recorded timestep. The numpy arrays (`obs`, `node_positions`) are
    transient — they are written to / read from sidecars, never into the manifest."""
    t: float = 0.0
    action: dict = field(default_factory=dict)      # {insertion, twist, aspiration}
    kinematics: dict = field(default_factory=dict)  # {tip_mm, tip_s, tip_r, max_r, node_positions_ref}
    obs_modality: str = "none"                      # "fluoro" | "luminal" | "none"
    obs_ref: str | None = None                      # observation sidecar filename (.npy)
    force: float | None = None                      # measured where instrumented; else None
    obs: object = field(default=None, repr=False, compare=False)             # transient array
    node_positions: object = field(default=None, repr=False, compare=False)  # transient (n,3)

    def to_dict(self) -> dict:                       # manifest entry: scalars + refs only
        return {"t": self.t, "action": self.action, "kinematics": self.kinematics,
                "obs_modality": self.obs_modality, "obs_ref": self.obs_ref, "force": self.force}

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(t=d.get("t", 0.0), action=d.get("action", {}),
                   kinematics=d.get("kinematics", {}), obs_modality=d.get("obs_modality", "none"),
                   obs_ref=d.get("obs_ref"), force=d.get("force"))

    def load_obs(self, root: str):
        """Lazy-load this step's observation sidecar from episode dir `root`."""
        if self.obs_ref is None:
            return None
        return np.load(os.path.join(root, "obs", self.obs_ref))

    def load_nodes(self, root: str):
        """Lazy-load this step's device node positions (n,3) if recorded."""
        ref = self.kinematics.get("node_positions_ref")
        return np.load(os.path.join(root, "obs", ref)) if ref else None


@dataclass
class Outcome:
    success: bool = False
    final_dist: float = 0.0
    steps: int = 0
    retrieval: str | None = None       # clot outcome where relevant (retrieve/slip/fragment)
    label: str = ""                    # free-form

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Outcome":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class Episode:
    meta: EpisodeMeta = field(default_factory=EpisodeMeta)
    steps: list = field(default_factory=list)        # list[Step]
    outcome: Outcome = field(default_factory=Outcome)

    def manifest(self) -> dict:
        # provenance at TOP LEVEL so the firewall (scans all *.json for a top-level
        # provenance key) covers episode manifests with no special-casing.
        return {"version": self.meta.version, "provenance": self.meta.provenance,
                "meta": self.meta.to_dict(), "steps": [s.to_dict() for s in self.steps],
                "outcome": self.outcome.to_dict()}

    def save(self, root: str) -> None:
        os.makedirs(os.path.join(root, "obs"), exist_ok=True)
        for s in self.steps:
            if s.obs is not None and s.obs_ref:
                np.save(os.path.join(root, "obs", s.obs_ref), np.asarray(s.obs))
            ref = s.kinematics.get("node_positions_ref")
            if s.node_positions is not None and ref:
                np.save(os.path.join(root, "obs", ref), np.asarray(s.node_positions))
        with open(os.path.join(root, "manifest.json"), "w") as f:
            json.dump(self.manifest(), f, indent=2)

    @classmethod
    def load(cls, root: str) -> "Episode":
        with open(os.path.join(root, "manifest.json")) as f:
            d = json.load(f)
        return cls(meta=EpisodeMeta.from_dict(d["meta"]),
                   steps=[Step.from_dict(s) for s in d["steps"]],
                   outcome=Outcome.from_dict(d["outcome"]))


def validate(ep: Episode) -> None:
    """Raise ValueError if the episode is malformed (used by capture + tests)."""
    if ep.meta.provenance not in ("procedural", "patient(private)"):
        raise ValueError(f"provenance must be 'procedural' or 'patient(private)', got {ep.meta.provenance!r}")
    if ep.meta.version != SCHEMA_VERSION:
        raise ValueError(f"version mismatch: expected {SCHEMA_VERSION}, got {ep.meta.version!r}")
    if not ep.steps:
        raise ValueError("episode has no steps")
    last_t = -np.inf
    for i, s in enumerate(ep.steps):
        if not np.isfinite(s.t):
            raise ValueError(f"step {i}: non-finite t")
        if s.t < last_t:
            raise ValueError(f"step {i}: time goes backwards ({s.t} < {last_t})")
        last_t = s.t
        tip = s.kinematics.get("tip_mm")
        if tip is not None and not np.all(np.isfinite(tip)):
            raise ValueError(f"step {i}: non-finite tip_mm {tip}")
        if s.obs_modality not in ("none", "fluoro", "luminal"):
            raise ValueError(f"step {i}: unknown obs_modality {s.obs_modality!r}")
    if not np.isfinite(ep.outcome.final_dist):
        raise ValueError("outcome.final_dist is non-finite")


if __name__ == "__main__":  # self-check: round-trip + validation
    import tempfile

    ep = Episode(
        meta=EpisodeMeta(asset_ref="straight.json", dt=5e-3, notes={"true_C10": 4000.0}),
        steps=[Step(t=i * 5e-3, action={"insertion": 1.0},
                    kinematics={"tip_mm": [0.0, 0.0, float(i)], "tip_s": float(i)},
                    obs_modality="fluoro", obs_ref=f"{i:03d}.npy", obs=np.full((4, 4), float(i)))
               for i in range(3)],
        outcome=Outcome(success=True, final_dist=0.4, steps=3, label="straight"))
    validate(ep)
    norm = lambda m: json.dumps(m, sort_keys=True)        # tuples<->lists via JSON, like on disk
    with tempfile.TemporaryDirectory() as d:
        ep.save(d)
        back = Episode.load(d)
        assert norm(back.manifest()) == norm(ep.manifest()), "manifest must round-trip"
        assert np.array_equal(back.steps[2].load_obs(d), np.full((4, 4), 2.0)), "sidecar must round-trip"
    bad = Episode(meta=EpisodeMeta(provenance="patient(private)"), steps=[Step()], outcome=Outcome())
    validate(bad)  # patient(private) is a VALID provenance value (the firewall, not validate, blocks commit)
    print("episode schema self-check ok")
