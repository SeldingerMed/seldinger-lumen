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
    <root>/<asset_ref>        optional self-contained lumen-asset/0 geometry
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


def _check_ref(ref: str, i: int, name: str) -> None:
    """Reject a sidecar ref that isn't a bare filename (path-traversal guard, H2)."""
    if os.path.isabs(ref) or os.sep in ref or "/" in ref or "\\" in ref or ".." in ref:
        raise ValueError(f"step {i}: {name} must be a bare filename, got {ref!r}")


def _safe_path(root: str, ref: str) -> str:
    """Resolve <root>/obs/<ref>, raising if it escapes the obs directory."""
    base = os.path.realpath(os.path.join(root, "obs"))
    full = os.path.realpath(os.path.join(base, ref))
    if os.path.commonpath([base, full]) != base:
        raise ValueError(f"sidecar ref escapes the obs directory: {ref!r}")
    return full


def _is_bare_file_ref(ref: str) -> bool:
    return bool(ref) and not (
        os.path.isabs(ref) or os.sep in ref or "/" in ref or "\\" in ref or ".." in ref
        or "://" in ref
    )


def _safe_root_file(root: str, ref: str) -> str:
    """Resolve <root>/<ref>, raising if a supposedly local file ref escapes root."""
    if not _is_bare_file_ref(ref):
        raise ValueError(f"local file ref must be a bare filename, got {ref!r}")
    base = os.path.realpath(root)
    full = os.path.realpath(os.path.join(base, ref))
    if os.path.commonpath([base, full]) != base:
        raise ValueError(f"local file ref escapes the episode directory: {ref!r}")
    return full


@dataclass
class EpisodeMeta:
    frame: Frame = field(default_factory=Frame)   # declared coordinate convention (reused)
    asset_ref: str = ""                           # path/id of the lumen-asset/0 geometry
    device: dict = field(default_factory=dict)    # device knobs (radius, stiffness, ...)
    sensor: dict = field(default_factory=dict)    # modality + render params
    calibration: dict = field(default_factory=dict)  # C-arm/scope calibration for replay
    labels: dict = field(default_factory=dict)     # task/anatomy/procedure labels
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
        # L3: unknown keys are dropped on purpose (forward-compat reads); the coarse
        # SCHEMA_VERSION pin in validate() is the gate against real cross-version skew.
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
        return np.load(_safe_path(root, self.obs_ref))      # escape-guarded (untrusted manifest)

    def load_nodes(self, root: str):
        """Lazy-load this step's device node positions (n,3) if recorded."""
        ref = self.kinematics.get("node_positions_ref")
        return np.load(_safe_path(root, ref)) if ref else None


@dataclass
class Outcome:
    success: bool = False
    final_dist: float = 0.0
    steps: int = 0
    retrieval: str | None = None       # clot outcome where relevant (retrieve/slip/fragment)
    label: str = ""                    # free-form
    metrics: dict = field(default_factory=dict)  # clinically meaningful endpoint summary

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
    asset: object = field(default=None, repr=False, compare=False)  # transient Asset sidecar

    def manifest(self) -> dict:
        # version/provenance are mirrored at TOP LEVEL (convenient for `jq`/grep and
        # consistent with the asset schema, whose provenance is top-level). meta.* is
        # canonical; load() checksums the two against each other so they can't drift.
        return {"version": self.meta.version, "provenance": self.meta.provenance,
                "meta": self.meta.to_dict(), "steps": [s.to_dict() for s in self.steps],
                "outcome": self.outcome.to_dict()}

    def save(self, root: str) -> None:
        validate(self)                                    # save is the declared enforcement gate
        obs_dir = os.path.join(root, "obs")
        os.makedirs(obs_dir, exist_ok=True)
        if self.asset is not None:
            if not self.meta.asset_ref:
                raise ValueError("episode asset set but meta.asset_ref missing")
            self.asset.save(_safe_root_file(root, self.meta.asset_ref))
        for i, s in enumerate(self.steps):
            if s.obs is not None:
                if not s.obs_ref:                         # don't silently drop data
                    raise ValueError(f"step {i}: obs set but obs_ref missing")
                np.save(_safe_path(root, s.obs_ref), np.asarray(s.obs))
            ref = s.kinematics.get("node_positions_ref")
            if s.node_positions is not None:
                if not ref:
                    raise ValueError(f"step {i}: node_positions set but node_positions_ref missing")
                np.save(_safe_path(root, ref), np.asarray(s.node_positions))
        # atomic manifest write: a crash mid-write must not leave an unloadable episode.
        tmp = os.path.join(root, "manifest.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.manifest(), f, indent=2)
        os.replace(tmp, os.path.join(root, "manifest.json"))
        # ponytail: stale sidecars from a prior save of a different episode aren't pruned;
        # add an obs/ clear if a corpus ever re-saves different episodes into one dir.

    @classmethod
    def load(cls, root: str) -> "Episode":
        # L7: load is intentionally UNCHECKED on the SEMANTIC contract (time/modality/
        # refs) so a malformed manifest can be loaded to inspect/repair; save() and the
        # corpus reader run validate(). The ONE structural thing load enforces is that
        # the top-level version/provenance mirror agrees with the canonical meta.* — a
        # disagreement means a hand-edited/merged manifest, and silently trusting either
        # copy would be a trust-boundary hole (e.g. top-level "procedural" hiding a
        # patient meta). meta is canonical; the mirror is a checksum.
        with open(os.path.join(root, "manifest.json")) as f:
            d = json.load(f)
        ep = cls(meta=EpisodeMeta.from_dict(d["meta"]),
                 steps=[Step.from_dict(s) for s in d["steps"]],
                 outcome=Outcome.from_dict(d["outcome"]))
        for key in ("version", "provenance"):
            top, nested = d.get(key), getattr(ep.meta, key)
            if top is not None and top != nested:
                raise ValueError(f"manifest top-level {key}={top!r} disagrees with "
                                 f"meta.{key}={nested!r} (tampered or merged manifest)")
        return ep

    def load_asset(self, root: str):
        """Load the episode-local lumen-asset sidecar, if asset_ref is a local file."""
        if not _is_bare_file_ref(self.meta.asset_ref):
            return None
        from lumen.assets.schema import Asset
        return Asset.load(_safe_root_file(root, self.meta.asset_ref))


def validate(ep: Episode, root: str | None = None) -> None:
    """Raise ValueError if the episode is malformed (used by save + capture + tests).

    The in-memory checks are cheap. Pass `root` (the episode dir) to also verify that
    every referenced sidecar file actually exists on disk."""
    if ep.meta.provenance not in ("procedural", "patient(private)"):
        raise ValueError(f"provenance must be 'procedural' or 'patient(private)', got {ep.meta.provenance!r}")
    if ep.meta.version != SCHEMA_VERSION:
        raise ValueError(f"version mismatch: expected {SCHEMA_VERSION}, got {ep.meta.version!r}")
    if not ep.steps:
        raise ValueError("episode has no steps")
    if ep.outcome.steps != len(ep.steps):
        raise ValueError(f"outcome.steps ({ep.outcome.steps}) != number of steps ({len(ep.steps)})")
    if not np.isfinite(ep.outcome.final_dist):
        raise ValueError("outcome.final_dist is non-finite")

    obs_refs, node_refs = [], []
    last_t = -np.inf
    for i, s in enumerate(ep.steps):
        if not np.isfinite(s.t):
            raise ValueError(f"step {i}: non-finite t")
        # L5: equal t is permitted on purpose (a held pose / sub-dt sample); only
        # time going BACKWARDS is malformed.
        if s.t < last_t:
            raise ValueError(f"step {i}: time goes backwards ({s.t} < {last_t})")
        last_t = s.t
        if s.obs_modality not in ("none", "fluoro", "luminal"):
            raise ValueError(f"step {i}: unknown obs_modality {s.obs_modality!r}")
        if s.obs_modality in ("fluoro", "luminal") and not s.obs_ref:
            raise ValueError(f"step {i}: obs_modality={s.obs_modality!r} requires obs_ref")
        if s.obs_modality == "none" and s.obs_ref:
            raise ValueError(f"step {i}: obs_modality='none' but obs_ref={s.obs_ref!r} set")
        if s.obs_ref:
            _check_ref(s.obs_ref, i, "obs_ref"); obs_refs.append(s.obs_ref)
        nref = s.kinematics.get("node_positions_ref")
        if nref:
            _check_ref(nref, i, "node_positions_ref"); node_refs.append(nref)
        tip = s.kinematics.get("tip_mm")
        if tip is not None:
            try:
                arr = np.asarray(tip, dtype=float)          # ValueError-contract even on garbage
            except (TypeError, ValueError) as e:
                raise ValueError(f"step {i}: tip_mm not numeric: {tip!r}") from e
            if arr.shape != (3,):
                raise ValueError(f"step {i}: tip_mm must be length-3, got {tip!r}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"step {i}: non-finite tip_mm {tip}")
    if len(set(obs_refs)) != len(obs_refs):                 # H1: a reused name clobbers earlier sidecars
        raise ValueError("duplicate obs_ref across steps (would clobber sidecars)")
    if len(set(node_refs)) != len(node_refs):
        raise ValueError("duplicate node_positions_ref across steps (would clobber sidecars)")
    if root is not None:                                    # closed-loop: referenced files must exist
        # Local asset refs make a captured episode self-contained. External/private ids
        # can still live in asset_ref, but bare filenames must resolve inside the episode.
        if _is_bare_file_ref(ep.meta.asset_ref):
            asset_path = _safe_root_file(root, ep.meta.asset_ref)
            if not os.path.exists(asset_path):
                raise ValueError(f"asset_ref sidecar missing on disk: {ep.meta.asset_ref}")
        for i, s in enumerate(ep.steps):
            if s.obs_ref and not os.path.exists(_safe_path(root, s.obs_ref)):
                raise ValueError(f"step {i}: obs_ref sidecar missing on disk: {s.obs_ref}")
            nref = s.kinematics.get("node_positions_ref")
            if nref and not os.path.exists(_safe_path(root, nref)):
                raise ValueError(f"step {i}: node_positions_ref sidecar missing on disk: {nref}")


if __name__ == "__main__":  # self-check: round-trip + validation
    import tempfile
    from lumen.assets import procedural

    def _ep(n=3):
        return Episode(
            meta=EpisodeMeta(asset_ref="straight.json", dt=5e-3, notes={"true_C10": 4000.0}),
            steps=[Step(t=i * 5e-3, action={"insertion": 1.0},
                        kinematics={"tip_mm": [0.0, 0.0, float(i)], "tip_s": float(i)},
                        obs_modality="fluoro", obs_ref=f"{i:03d}.npy", obs=np.full((4, 4), float(i)))
                   for i in range(n)],
            outcome=Outcome(success=True, final_dist=0.4, steps=n, label="straight"),
            asset=procedural.straight_tube(80.0, 2.0))

    ep = _ep()
    validate(ep)
    norm = lambda m: json.dumps(m, sort_keys=True)        # tuples<->lists via JSON, like on disk
    with tempfile.TemporaryDirectory() as d:
        ep.save(d)
        back = Episode.load(d)
        assert norm(back.manifest()) == norm(ep.manifest()), "manifest must round-trip"
        assert np.array_equal(back.steps[2].load_obs(d), np.full((4, 4), 2.0)), "sidecar must round-trip"
        validate(back, root=d)                            # root mode: every sidecar exists

    def _rejects(ep, why):
        try:
            validate(ep)
        except ValueError:
            return
        raise AssertionError(f"validate should reject: {why}")

    dup = _ep(2); dup.steps[1].obs_ref = dup.steps[0].obs_ref
    _rejects(dup, "duplicate obs_ref clobbers sidecars")
    eviln = _ep(1); eviln.steps[0].obs_ref = "../evil.npy"
    _rejects(eviln, "path-traversal obs_ref")
    noref = _ep(1); noref.steps[0].obs_ref = None
    _rejects(noref, "fluoro step without obs_ref")
    validate(Episode(meta=EpisodeMeta(provenance="patient(private)"), steps=[Step()],
                     outcome=Outcome(steps=1)))           # patient(private) is a VALID value (firewall blocks commit)
    print("episode schema self-check ok")
