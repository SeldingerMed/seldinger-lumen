"""E2-compatible public replacement corpus, lineage audit and frozen holdout.

This is a *replacement* module (not the original E2 pipeline): it builds a
small, fully synthetic, frozen public corpus from actual CPU ``NavEnv``
rollouts (5-state observations, 2-action schedules), stratifies degradation by
named lineage, detects trace lineage at exact and near content distances,
literal shared observation frames (including time-shifted alignment), shared
seed/anatomy+action relationships and declared parent siblings, groups lineage
transitively, freezes the held-out family behind a digest (independence is
*verified* against every full transitive group, never assumed), and reruns the
original fixed action-conditioned ridge baseline on the corpus separately for
the dev and held-out families, with per-heldout-trace metrics suppressed when
the holdout audit fails.

Design corrections over the earlier draft:

* Content digests hash ONLY the raw observations+actions — never trace ids,
  splits, lineage names or other metadata, and with NO quantization: identical
  content means an exactly identical digest.
* ``near`` is caught by raw observation distance regardless of metadata.
* ``seed_sibling`` fires for any shared generator+seed pair, including same
  geometry with different actions; declared parent siblings are caught by
  explicit provenance.
* Frame sharing is literal: a frame is shared when every scalar matches within
  ``FRAME_TOL``, allowing time-shifted alignment between the traces.
* ``anatomy_action`` means *shared anatomy and shared actions with different
  observations*.
* Comparisons use the full raw 5-state observation distance, not a single
  scalar curve.
* The frozen holdout is derived from the full transitive grouping: it is
  reported independent only when development is nonempty and zero groups cross
  families. Crossing groups are reported, never filtered out retrospectively.

No real E2 data is read or embedded anywhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA = "lumen-lineage-audit/2"
HOLDOUT_SCHEMA = "lumen-lineage-holdout/2"
CORPUS_SCHEMA = "lumen-e2-compatible-corpus/1"

STRATA = ("exact", "near", "frame_shared", "anatomy_action", "declared",
          "seed_sibling")
NEAR_TOL = 1e-6      # raw observation distance treated as a near duplicate
FRAME_TOL = 1e-3     # per-scalar tolerance for literal shared-frame detection
FRAME_OVERLAP = 0.5  # required fraction of shared frames (at best shift)
ACTION_ATOL = 1e-9   # actions compared with this tolerance

# Deterministic offsets applied to the recorded observations of declared
# degradation variants (applied identically to every scalar, so they are
# fully reproducible).
DEGRADATION_OFFSETS = {"near": 2e-7, "frame": 1.5e-4, "perturbed": 2e-2}

# Frozen action schedules: fixed (insertion, twist) pairs, 12 steps each.
SCHEDULES = {
    "A": ((1.0, 0.0), (0.8, 0.2), (1.0, -0.2), (0.5, 0.0), (1.0, 0.1),
          (0.0, 0.0), (0.7, 0.3), (1.0, 0.0), (0.9, 0.0), (1.0, 0.0),
          (0.6, -0.1), (1.0, 0.0)),
    "B": ((0.9, 0.1), (1.0, 0.0), (0.4, -0.2), (1.0, 0.2), (0.8, 0.0),
          (1.0, 0.0), (0.3, 0.1), (1.0, -0.1), (0.7, 0.0), (1.0, 0.3),
          (0.9, 0.0), (0.5, 0.0)),
    "C": ((0.5, 0.0), (1.0, 0.3), (0.2, -0.1), (1.0, 0.0), (0.9, 0.2),
          (0.6, 0.0), (1.0, 0.0), (0.8, -0.3), (1.0, 0.1), (0.4, 0.0),
          (1.0, 0.0), (0.9, 0.0)),
    "D": ((1.0, 0.0), (0.6, 0.1), (0.9, -0.2), (0.3, 0.0), (1.0, 0.2),
          (0.8, 0.0), (1.0, -0.1), (0.5, 0.3), (1.0, 0.0), (0.7, 0.0),
          (1.0, 0.1), (0.9, 0.0)),
    "E": ((0.8, 0.2), (1.0, 0.0), (0.6, -0.1), (1.0, 0.1), (0.4, 0.0),
          (1.0, -0.2), (0.9, 0.0), (1.0, 0.3), (0.5, 0.0), (1.0, 0.0),
          (0.7, 0.2), (1.0, 0.0)),
}

# Frozen corpus plan: family -> list of base lineage specs.
# Dev family: stenotic geometry, seeds 10-12, schedules A-C; the fourth base
# deliberately shares seed 10 + schedule A with the first under a different
# anatomy, exercising the seed-sibling stratum.
# Held-out family: tortuous geometry, seeds 90/91, schedules D/E — nothing in
# the dev family shares geometry, seed or schedule with it. Both families get
# the full clean + declared-variant set so held-out rows are stratified by
# actual perturbation.
_BASES = {
    "dev": [
        ("dev/sev0.5/seed10", "stenotic_tube", 0.5, 10, "A"),
        ("dev/sev0.3/seed11", "stenotic_tube", 0.3, 11, "B"),
        ("dev/sev0.7/seed12", "stenotic_tube", 0.7, 12, "C"),
        ("dev/tort0.3/seed10", "tortuous_tube", 0.3, 10, "A"),
    ],
    "heldout": [
        ("heldout/tort0.35/seed90", "tortuous_tube", 0.35, 90, "D"),
        ("heldout/tort0.55/seed91", "tortuous_tube", 0.55, 91, "E"),
    ],
}
_VARIANT_SPLITS = {"clean": "train", "near": "train", "frame": "train",
                   "perturbed": "train", "exact": "test"}


# --------------------------------------------------------------- digests


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------- traces


@dataclass
class TrajectoryTrace:
    """A producer trajectory reduced to its observation/action record.

    ``content_sha256`` is a digest of the RAW observations and actions ONLY —
    no quantization, and deliberately independent of every metadata field
    (ids, splits, lineage names, anatomy, seeds). Identical content therefore
    always means an identical digest.
    """

    trace_id: str
    split: str
    family: str           # "dev" or "heldout"
    lineage: str          # named lineage this trace belongs to
    generator: str
    seed: int
    anatomy: dict         # generator parameters (base geometry)
    schedule_id: str
    actions: tuple        # tuple of per-step (insertion, twist) pairs
    observations: tuple   # tuple of per-step 5-state observation tuples
    degradation: str = "clean"   # clean | near | frame | perturbed | exact
    parent: str | None = None    # trace_id of the base this variant derives from
    provenance: dict = field(default_factory=dict)

    @property
    def content_sha256(self) -> str:
        return _digest({
            "actions": [[float(v) for v in a] for a in self.actions],
            "observations": [[float(v) for v in o] for o in self.observations],
        })

    def anatomy_key(self) -> str:
        return _digest({"generator": self.generator,
                        "anatomy": _jsonable(self.anatomy)})

    def records(self) -> list[dict]:
        """Per-step records in the original baseline's JSON transition format:
        ``[{"obs": [...], "action": [...]}, ...]`` where consecutive records
        form (obs_t, action_t) -> obs_{t+1} transitions."""
        return [{"obs": [float(v) for v in o], "action": [float(v) for v in a]}
                for o, a in zip(self.observations, self.actions)]


def producer_trace(seed: int, *, env_factory=None, split: str, family: str,
                   lineage: str, generator: str, anatomy: dict,
                   schedule_id: str, action_schedule,
                   trace_id: str | None = None, degradation: str = "clean",
                   parent: str | None = None,
                   obs_offset: float = 0.0) -> TrajectoryTrace:
    """Roll an actual CPU NavEnv episode with a fixed 2-action schedule.

    Deterministic: identical factory, seed and schedule always reproduce the
    identical content digest. ``obs_offset`` applies a declared observation
    perturbation to the *recorded* observations only (the environment
    trajectory itself is unchanged). With ``env_factory=None`` a stenotic
    NavEnv is built from ``anatomy``."""
    if env_factory is None:
        env_factory = _env_factory(generator, anatomy, len(action_schedule))
    env = env_factory()
    obs, _ = env.reset(seed=int(seed))
    actions: list[tuple[float, float]] = []
    observations: list[tuple[float, ...]] = []
    for a, w in action_schedule:
        act = np.asarray([float(a), float(w)], dtype=float)
        actions.append((float(act[0]), float(act[1])))
        o = np.asarray(obs, dtype=float)
        if obs_offset:
            o = o + obs_offset
        observations.append(tuple(float(v) for v in o))
        obs, _r, terminated, truncated, _info = env.step(act)
        if terminated or truncated:
            break
    if trace_id is None:
        trace_id = f"{lineage}/{degradation}"
    return TrajectoryTrace(
        trace_id=trace_id, split=split, family=family, lineage=lineage,
        generator=generator, seed=int(seed), anatomy=_jsonable(anatomy),
        schedule_id=schedule_id, actions=tuple(actions),
        observations=tuple(observations), degradation=degradation,
        parent=parent,
        provenance={"source": "navenv_cpu_rollout", "steps": len(actions),
                    "obs_offset": float(obs_offset)},
    )


def _env_factory(generator: str, anatomy: dict, n_steps: int):
    from lumen.envs import registration

    def factory():
        if anatomy.get("asset") == "tortuous_tube":
            return registration.make_nav_tortuous(
                severity=float(anatomy["severity"]), max_steps=n_steps, device="cpu")
        return registration.make_nav_stenotic(
            severity=float(anatomy["severity"]), max_steps=n_steps, device="cpu")
    return factory


def _clone(trace: TrajectoryTrace, *, trace_id: str, lineage: str,
           split: str, degradation: str, parent: str | None,
           obs_offset: float = 0.0) -> TrajectoryTrace:
    """Derive a variant of ``trace`` without re-rolling the environment.

    Sibling variants keep the base's seed, anatomy, action schedule and
    (offset) observations so the whole sibling family stays together."""
    offset = float(obs_offset)
    observations = tuple(
        tuple(float(v) + offset for v in o) for o in trace.observations)
    return TrajectoryTrace(
        trace_id=trace_id, split=split, family=trace.family, lineage=lineage,
        generator=trace.generator, seed=trace.seed, anatomy=dict(trace.anatomy),
        schedule_id=trace.schedule_id, actions=trace.actions,
        observations=observations, degradation=degradation, parent=parent,
        provenance={"source": "declared_variant", "obs_offset": offset,
                    "copied_from": trace.trace_id},
    )


def build_corpus() -> tuple[list[TrajectoryTrace], list[TrajectoryTrace], dict]:
    """Build the frozen E2-compatible replacement corpus.

    Returns ``(dev_traces, heldout_traces, meta)``. EVERY base — dev and
    held-out alike — gets a clean trace plus declared near / frame / perturbed
    observation variants (kept on the base's split) and an exact content
    duplicate, so held-out rows exist for every actual degradation level. The
    held-out family uses distinct geometry, seeds and action schedules and is
    frozen behind a digest.
    """
    traces: dict[str, list[TrajectoryTrace]] = {"dev": [], "heldout": []}
    sibling_variants: dict[str, dict[str, str]] = {}
    for family, bases in _BASES.items():
        for lineage, asset, severity, seed, sched_id in bases:
            schedule = SCHEDULES[sched_id]
            anatomy = {"asset": asset, "severity": float(severity)}
            factory = _env_factory("nav", anatomy, len(schedule))
            base_split = "heldout" if family == "heldout" else "train"
            base = producer_trace(
                seed,
                env_factory=factory,
                split=base_split,
                family=family,
                lineage=lineage, generator="nav", anatomy=anatomy,
                schedule_id=sched_id, action_schedule=schedule,
                trace_id=f"{lineage}/clean")
            traces[family].append(base)
            variants: dict[str, str] = {}
            for degradation in ("near", "frame", "perturbed"):
                v = _clone(
                    base,
                    trace_id=f"{lineage}/{degradation}",
                    lineage=f"{lineage}#{degradation}",
                    split=_VARIANT_SPLITS[degradation]
                    if family == "dev" else base_split,
                    degradation=degradation, parent=base.trace_id,
                    obs_offset=DEGRADATION_OFFSETS[degradation])
                traces[family].append(v)
                variants[v.lineage] = degradation
            # exact content duplicate (declared lineage probe; different
            # metadata, identical content). Dev puts it on the test split;
            # the held-out family keeps it on the heldout split.
            dup = _clone(base, trace_id=f"{lineage}/exactdup",
                         lineage=f"{lineage}#exact",
                         split=_VARIANT_SPLITS["exact"]
                         if family == "dev" else base_split,
                         degradation="exact",
                         parent=base.trace_id)
            traces[family].append(dup)
            variants[dup.lineage] = "exact"
            sibling_variants[base.lineage] = variants
    meta = {
        "schema": CORPUS_SCHEMA,
        "e2_compatible": True,
        "data_provenance": "synthetic procedural NavEnv rollouts",
        "action_space": 2,
        "observation_space": 5,
        "schedule_ids": sorted(SCHEDULES),
        "sibling_variants": sibling_variants,
        "holdout_lineages": sorted(lineage for lineage, _, _, _, _
                                   in _BASES["heldout"]),
        "declared_holdout_digest": _digest({
            "schema": HOLDOUT_SCHEMA,
            "lineages": sorted(l for l, _, _, _, _ in _BASES["heldout"]),
            "seeds": sorted(s for _, _, _, s, _ in _BASES["heldout"]),
        }),
        "declared_degradation_counts": {
            family: {
                degradation: sum(
                    1 for t in traces[family] if t.degradation == degradation)
                for degradation in ("clean", "near", "frame", "perturbed",
                                    "exact")
            } for family in ("dev", "heldout")
        },
    }
    return traces["dev"], traces["heldout"], meta


def leaky_control(traces, rng_seed: int = 0):
    """Deliberately leaky negative control: clone one trace into another
    split under a fresh ``leak/`` id. Returns (augmented, injected_ids)."""
    if len(traces) < 2:
        raise ValueError("need at least two traces to inject a cross-split leak")
    splits = sorted({t.split for t in traces})
    if len(splits) < 2:
        raise ValueError("need at least two distinct splits")
    order = sorted(range(len(traces)), key=lambda i: traces[i].content_sha256)
    rng = np.random.default_rng(rng_seed)
    src = traces[order[int(rng.integers(len(order)))]]
    target_split = sorted(set(splits) - {src.split})[
        int(rng.integers(len(splits) - 1))]
    clone = _clone(src, trace_id=f"leak/{src.content_sha256[:12]}/{target_split}",
                   lineage=f"leak/{src.lineage}", split=target_split,
                   degradation=src.degradation, parent=src.trace_id)
    return list(traces) + [clone], sorted([src.trace_id, clone.trace_id])


# -------------------------------------------------------------- matching


def _max_obs_dev(a: TrajectoryTrace, b: TrajectoryTrace) -> float:
    """Raw maximum absolute observation distance (full 5-state vectors)."""
    if len(a.observations) != len(b.observations) or not a.observations:
        return float("inf")
    return float(np.max(np.abs(
        np.asarray(a.observations, dtype=float)
        - np.asarray(b.observations, dtype=float))))


def _actions_match(a: TrajectoryTrace, b: TrajectoryTrace) -> bool:
    if len(a.actions) != len(b.actions):
        return False
    return all(np.allclose(x, y, atol=ACTION_ATOL, rtol=0.0)
               for x, y in zip(a.actions, b.actions))


def _frame_alignment(a: TrajectoryTrace, b: TrajectoryTrace,
                     tol: float) -> tuple[int, int, int]:
    """Literal shared-frame alignment, allowing a time shift.

    A frame is *shared* when every one of its 5 scalars matches within
    ``tol``. Returns ``(shared_frames, overlap_len, shift)`` for the shift
    with the most shared frames, where ``b[i + shift]`` aligns with ``a[i]``.
    """
    la, lb = len(a.observations), len(b.observations)
    if la == 0 or lb == 0:
        return 0, 0, 0
    oa = np.asarray(a.observations, dtype=float)
    ob = np.asarray(b.observations, dtype=float)
    best = (0, 0, 0)
    for shift in range(-(lb - 1), la):
        lo = max(0, -shift)
        hi = min(la, lb - shift)
        if hi <= lo:
            continue
        diff = np.abs(oa[lo:hi] - ob[lo + shift:hi + shift])
        shared = int(np.sum(np.max(diff, axis=1) <= tol))
        denom = hi - lo
        if (shared, denom) > (best[0], best[1]):
            best = (shared, denom, shift)
    return best


def _pair_kind(a: TrajectoryTrace, b: TrajectoryTrace, *, near_tol: float,
               frame_tol: float, frame_overlap: float) -> str | None:
    if a.trace_id == b.trace_id:
        return None
    if a.content_sha256 == b.content_sha256:
        return "exact"                      # metadata-independent digest
    # near is caught by raw observation distance regardless of any metadata
    if _max_obs_dev(a, b) <= near_tol:
        return "near"
    same_actions = _actions_match(a, b)
    same_anatomy = a.anatomy_key() == b.anatomy_key()
    if same_actions and same_anatomy:
        shared, denom, _shift = _frame_alignment(a, b, frame_tol)
        if denom and shared / denom >= frame_overlap:
            return "frame_shared"           # literal shared frames (any shift)
        return "anatomy_action"             # shared anatomy+actions, obs differ
    # declared provenance: explicit parent/child or full siblings
    if (a.parent == b.trace_id or b.parent == a.trace_id
            or (a.parent is not None and a.parent == b.parent)):
        return "declared"
    # shared generator+seed, even with identical geometry but different actions
    if a.generator == b.generator and a.seed == b.seed:
        return "seed_sibling"
    return None


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


# ----------------------------------------------------------------- audit


def audit(traces, *, near_tol: float = NEAR_TOL, frame_tol: float = FRAME_TOL,
          frame_overlap: float = FRAME_OVERLAP) -> dict:
    """Full lineage audit. JSON-serializable; integrates with the parent.

    Every reported pair carries ``max_obs_dev`` (maximum absolute observation
    distance, null when lengths differ) and ``shared_frames`` (count of
    literally shared frames at the best time shift). Degradation is
    stratified by the traces' ACTUAL perturbation labels, not by match kind.
    """
    traces = list(traces)
    by_id = {t.trace_id: t for t in traces}
    if len(by_id) != len(traces):
        raise ValueError("duplicate trace_id")

    pairs: list[dict] = []
    uf = _UnionFind()
    for t in traces:
        uf.find(t.trace_id)
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            a, b = traces[i], traces[j]
            kind = _pair_kind(a, b, near_tol=near_tol, frame_tol=frame_tol,
                              frame_overlap=frame_overlap)
            if kind is not None:
                uf.union(a.trace_id, b.trace_id)
                dev = _max_obs_dev(a, b)
                shared, _denom, shift = _frame_alignment(a, b, frame_tol)
                pairs.append({
                    "kind": kind, "a": a.trace_id, "b": b.trace_id,
                    "splits": sorted({a.split, b.split}),
                    "cross_split": a.split != b.split,
                    "max_obs_dev": dev if np.isfinite(dev) else None,
                    "shared_frames": shared,
                    "frame_shift": shift,
                    "degradations": sorted({a.degradation, b.degradation}),
                })

    members: dict[str, list[str]] = {}
    for t in traces:
        members.setdefault(uf.find(t.trace_id), []).append(t.trace_id)

    groups = []
    for root in sorted(members):
        ids = sorted(members[root])
        groups.append({
            "group_id": _digest({"members": ids})[:16],
            "trajectories": ids,
            "lineages": sorted({by_id[i].lineage for i in ids}),
            "splits": sorted({by_id[i].split for i in ids}),
            "families": sorted({by_id[i].family for i in ids}),
            "match_kinds": sorted({p["kind"] for p in pairs
                                   if p["a"] in ids or p["b"] in ids}),
            "cross_split": len({by_id[i].split for i in ids}) > 1,
        })
    group_of = {tid: g["group_id"] for g in groups for tid in g["trajectories"]}

    strata = {s: sum(1 for p in pairs if p["kind"] == s) for s in STRATA}
    by_degradation: dict[str, int] = {}
    for p in pairs:
        key = "+".join(p["degradations"])
        by_degradation[key] = by_degradation.get(key, 0) + 1

    return {
        "schema": SCHEMA,
        "counts": {
            "trajectories": len(traces),
            "pairs_compared": len(traces) * (len(traces) - 1) // 2,
            "matched_pairs": len(pairs),
            "groups": len(groups),
            "cross_split_groups": sum(1 for g in groups if g["cross_split"]),
            "by_kind": strata,
        },
        "pairs": pairs,
        "groups": groups,
        "lineage": {tid: {"group_id": group_of.get(tid),
                          "lineage": by_id[tid].lineage,
                          "family": by_id[tid].family,
                          "degradation": by_id[tid].degradation,
                          "links": sorted(
                              {p["a"] if p["b"] == tid else p["b"]
                               for p in pairs if tid in (p["a"], p["b"])})}
                    for tid, t in sorted(by_id.items())},
        "degradation": {
            "strata": STRATA,
            "counts": strata,
            "by_degradation": dict(sorted(by_degradation.items())),
            "worst_observed": next((s for s in STRATA if strata[s]), None),
        },
        "provenance": {tid: {
            "split": t.split, "family": t.family, "lineage": t.lineage,
            "generator": t.generator, "seed": t.seed, "anatomy": t.anatomy,
            "schedule_id": t.schedule_id, "degradation": t.degradation,
            "parent": t.parent, "content_sha256": t.content_sha256,
            "steps": len(t.actions), "provenance": t.provenance,
        } for tid, t in sorted(by_id.items())},
        "heldout": _frozen_holdout(traces, groups),
    }


def _frozen_holdout(traces, groups) -> dict:
    """Frozen held-out family derived from EVERY full transitive group.

    A group belongs to the holdout only when all of its members come from the
    ``heldout`` family. Independence is verified, never assumed: it holds only
    when the development family is nonempty AND zero groups contain members of
    both families. Crossing groups are reported, never filtered out
    retrospectively, and the frozen digest covers the exact held-out group
    membership so any membership change invalidates it.
    """
    by_id = {t.trace_id: t for t in traces}
    dev_nonempty = any(t.family == "dev" for t in traces)
    held_groups = []
    crossing = []
    for g in groups:
        fams = {by_id[i].family for i in g["trajectories"]}
        if fams == {"heldout"}:
            held_groups.append(g)
        elif len(fams) > 1:
            crossing.append(g["group_id"])
    independent = dev_nonempty and not crossing
    return {
        "schema": HOLDOUT_SCHEMA,
        "groups": [{"group_id": g["group_id"], "lineages": g["lineages"],
                    "trajectories": g["trajectories"], "family": "heldout"}
                   for g in held_groups],
        "crossing_groups": sorted(crossing),
        "development_nonempty": dev_nonempty,
        "independent": independent,
        "complete": bool(dev_nonempty and held_groups),
        "frozen_digest": _digest({
            "schema": HOLDOUT_SCHEMA,
            "groups": [(g["group_id"], g["trajectories"]) for g in held_groups],
        }),
    }


def audit_with_negative_control(traces, *, inject: bool = True,
                                rng_seed: int = 0, **kwargs) -> dict:
    """audit() plus the deliberately leaky control.

    With ``inject=True`` a cross-split duplicate is injected and the detector
    must flag exactly the injected pair (with at least one exact match) —
    otherwise the control fails. With ``inject=False`` it is the non-vacuous
    clean run: nothing is injected, so any detection is a false positive and
    fails the control.
    """
    injected: list[str] = []
    if inject:
        traces, injected = leaky_control(traces, rng_seed=rng_seed)
    result = audit(traces, **kwargs)
    if inject:
        matches = [p for p in result["pairs"] if p["cross_split"] and p["kind"] == "exact"
                   and sorted([p["a"], p["b"]]) == injected]
        detected = injected if matches else []
        passed = bool(matches)
    else:
        detected = sorted({tid for p in result["pairs"] if p["cross_split"]
                           for tid in (p["a"], p["b"])})
        passed = not detected
    result["negative_control"] = {
        "injected": sorted(injected), "detected": detected, "passed": passed,
    }
    return result


# ------------------------------------------------- fixed ridge baseline


def _load_transitions(traces) -> list[tuple]:
    """Same transition construction as the original baseline: consecutive
    records give (obs_t, action_t) -> obs_{t+1}."""
    rows = []
    for trace in traces:
        records = trace.records()
        for left, right in zip(records, records[1:]):
            obs = np.asarray(left["obs"], dtype=float)
            action = np.asarray(left["action"], dtype=float)
            target = np.asarray(right["obs"], dtype=float)
            rows.append((np.concatenate([obs, action]), obs, target))
    return rows


def run_baseline(train_traces, test_traces, ridge: float = 1e-4) -> dict:
    """The original fixed action-conditioned ridge baseline, rerun on this
    corpus (identical formulas, schema and decision rule). No evidence grade
    is attached — grading happens outside this function."""
    train = _load_transitions(train_traces)
    test = _load_transitions(test_traces)
    if not train or not test:
        raise ValueError("nonempty disjoint traces required")
    x = np.stack([row[0] for row in train])
    y = np.stack([row[2] for row in train])
    ones = np.ones((x.shape[0], 1))
    design = np.concatenate([x, ones], axis=1)
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ y)
    tx = np.stack([row[0] for row in test])
    current = np.stack([row[1] for row in test])
    target = np.stack([row[2] for row in test])
    pred = np.concatenate([tx, np.ones((tx.shape[0], 1))], axis=1) @ weights
    persistence = float(np.mean((current - target) ** 2))
    action_mse = float(np.mean((pred - target) ** 2))
    train_pred = design @ weights
    residual = np.sqrt(np.mean((train_pred - y) ** 2, axis=1))
    threshold = float(np.quantile(residual, 0.90))
    test_residual = np.sqrt(np.mean((pred - target) ** 2, axis=1))
    abstain = test_residual > threshold
    return {
        "schema": "seldinger-world-model-baseline-v1",
        "train_transitions": len(train),
        "test_transitions": len(test),
        "persistence_mse": persistence,
        "action_conditioned_mse": action_mse,
        "relative_change": (persistence - action_mse) / persistence
        if persistence else None,
        "calibration": {
            "train_residual_q90": threshold,
            "test_coverage_at_q90": float(np.mean(test_residual <= threshold)),
            "retrospective_abstention_rate": float(np.mean(abstain)),
            "warning": "retrospective residual abstention is diagnostic, "
                       "not deployable uncertainty",
        },
        "decision": "continue" if action_mse < persistence
        else "negative result; change model before follow-on",
    }


# ------------------------------------------------------- replacement API


def evaluate_heldout(train_traces, heldout_traces, frozen: dict) -> dict:
    """Per-heldout-trace fixed-baseline metrics, stratified by each trace's
    actual perturbation label. Metrics are SUPPRESSED entirely when the
    frozen holdout audit failed (not independent, incomplete, or empty)."""
    if not (frozen.get("independent") and frozen.get("groups")):
        return {"suppressed": True,
                "reason": "frozen holdout audit failed: holdout is not "
                          "independent or is empty; no metrics reported"}
    traces_metrics: dict[str, dict] = {}
    for t in heldout_traces:
        m = run_baseline(train_traces, [t])
        m["lineage"] = t.lineage
        m["degradation"] = t.degradation
        traces_metrics[t.trace_id] = m
    return {"suppressed": False, "traces": traces_metrics}


def replacement_audit() -> dict:
    """One-shot entry point: build the frozen corpus, audit lineage,
    run the leaky negative control, and rerun the original fixed baseline
    separately on the dev and held-out families. Held-out metrics are
    per-trace and stratified by actual perturbation; they are suppressed
    when the holdout audit fails. Primary training uses clean dev traces
    only — planted dev copies (variants and exact duplicates) never enter
    the training set. Returns a JSON dict."""
    dev, heldout, meta = build_corpus()
    result = audit(dev + heldout)
    control = audit_with_negative_control(
        [t for t in dev + heldout if t.degradation == "clean"], inject=True)
    train = [t for t in dev if t.split == "train"
             and t.degradation == "clean"]
    test = [t for t in dev if t.split == "test"]
    return {
        "schema": SCHEMA,
        "verdict": "independent" if result["heldout"]["independent"] else "leakage_downgraded",
        "claim_ceiling": "Replacement holdout only; perturbations are not independent episodes.",
        "corpus": meta,
        "audit": result,
        "negative_control": {
            "injected": control["negative_control"]["injected"],
            "detected": control["negative_control"]["detected"],
            "passed": control["negative_control"]["passed"],
        },
        "baseline": {
            "dev": {**run_baseline(train, test), "role": "positive_control_exact_content"},
            "heldout": evaluate_heldout(train, heldout, result["heldout"]),
        },
        "provenance": {
            "data": "synthetic procedural NavEnv rollouts",
            "real_e2_data_used": False,
        },
    }
