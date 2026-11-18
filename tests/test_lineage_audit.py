"""Tests for the E2-compatible public replacement corpus, lineage audit and
frozen holdout tracer.

Everything here is synthetic and procedural: actual CPU NavEnv rollouts with
fixed action schedules, deterministic clean and declared perturbed observation
variants on BOTH families, and a frozen held-out family with distinct
geometry/seeds/schedules whose independence is verified against every full
transitive group. No real E2 data is used anywhere.
"""

import dataclasses
import json

import numpy as np
import pytest

from lumen.experiments.lineage_audit import (
    STRATA,
    TrajectoryTrace,
    audit,
    audit_with_negative_control,
    build_corpus,
    evaluate_heldout,
    leaky_control,
    producer_trace,
    replacement_audit,
    run_baseline,
)

# ---------------------------------------------------------------- corpus


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


def test_corpus_families_are_frozen_and_distinct(corpus):
    dev, heldout, meta = corpus
    assert dev and heldout
    assert {t.family for t in dev} == {"dev"}
    assert {t.family for t in heldout} == {"heldout"}
    # held-out family: distinct geometry, seeds and action schedules
    dev_keys = {(t.generator, t.seed, t.schedule_id) for t in dev}
    held_keys = {(t.generator, t.seed, t.schedule_id) for t in heldout}
    assert not (dev_keys & held_keys)
    assert meta["frozen_holdout_digest"]
    assert meta["holdout_lineages"]


def test_corpus_observation_shapes_are_5state_2action(corpus):
    dev, heldout, _ = corpus
    for t in dev + heldout:
        assert len(t.observations) == len(t.actions) > 0
        assert all(len(o) == 5 for o in t.observations)
        assert all(len(a) == 2 for a in t.actions)


def test_heldout_gets_full_degradation_stratification(corpus):
    """Held-out rows exist for every ACTUAL perturbation level, on both bases."""
    dev, heldout, meta = corpus
    assert len(heldout) == 10  # 2 bases x (clean + near + frame + perturbed + exact)
    held_degradations = {t.degradation for t in heldout}
    assert held_degradations == {"clean", "near", "frame", "perturbed", "exact"}
    held_bases = {t.lineage for t in heldout if t.degradation == "clean"}
    assert held_bases == set(meta["holdout_lineages"])
    for base_lineage in held_bases:
        derived = {t.lineage for t in heldout if t.lineage.startswith(base_lineage)}
        assert len(derived) == 5
    # declared degradation counts match reality for both families
    for family, traces in (("dev", dev), ("heldout", heldout)):
        counts = meta["declared_degradation_counts"][family]
        for degradation, count in counts.items():
            assert count == sum(1 for t in traces
                                if t.degradation == degradation)


def test_corpus_sibling_variants_stay_with_their_base(corpus):
    dev, heldout, meta = corpus
    by_lineage = {t.lineage: t for t in dev + heldout}
    for base_lineage, variants in meta["sibling_variants"].items():
        base = by_lineage[base_lineage]
        for v_lineage, v_degradation in variants.items():
            v = by_lineage[v_lineage]
            if v_degradation == "exact" and base.family == "dev":
                # the declared exact duplicate intentionally probes the dev
                # test split; held-out duplicates stay on the heldout split
                assert v.split == "test"
            else:
                assert v.split == base.split
            assert v.family == base.family
            assert v.parent == base.trace_id
            assert v.degradation == v_degradation
            # same seed + anatomy + action schedule as the base
            assert v.seed == base.seed
            assert v.anatomy == base.anatomy
            assert v.actions == base.actions


def test_corpus_declared_degradation_offsets(corpus):
    """Each base carries a clean observation plus declared near / frame /
    perturbed observation variants, all deterministic."""
    dev, heldout, meta = corpus
    by_lineage = {t.lineage: t for t in dev + heldout}
    offsets = {"near": 2e-7, "frame": 1.5e-4, "perturbed": 2e-2}
    for base_lineage, variants in meta["sibling_variants"].items():
        base = by_lineage[base_lineage]
        for v_lineage, degradation in variants.items():
            if degradation == "exact":
                continue  # content-identical clone, no offset
            v = by_lineage[v_lineage]
            delta = np.asarray(v.observations) - np.asarray(base.observations)
            assert np.allclose(delta, offsets[degradation], atol=1e-12)
    dev2, heldout2, _ = build_corpus()
    by_lineage2 = {t.lineage: t for t in dev2 + heldout2}
    for lineage, t in by_lineage.items():
        assert by_lineage2[lineage].content_sha256 == t.content_sha256


def test_corpus_covers_content_behavioral_strata(corpus):
    dev, heldout, _ = corpus
    result = audit(dev + heldout)
    counts = result["degradation"]["counts"]
    for stratum in ("exact", "near", "frame_shared", "anatomy_action",
                    "seed_sibling"):
        assert counts[stratum] > 0, stratum


def test_audit_degradation_stratified_by_actual_perturbation(corpus):
    dev, heldout, _ = corpus
    result = audit(dev + heldout)
    by_deg = result["degradation"]["by_degradation"]
    assert by_deg, "actual-perturbation stratification must be reported"
    for key in by_deg:
        labels = set(key.split("+"))
        assert labels <= {"clean", "near", "frame", "perturbed", "exact"}


# ------------------------------------------------------- content digests


def test_content_hash_is_independent_of_metadata():
    a = TrajectoryTrace(
        trace_id="x", split="train", family="dev", lineage="L", generator="g",
        seed=1, anatomy={"severity": 0.5}, schedule_id="A",
        actions=((1.0, 0.0),), observations=((0.1, 0.2, 0.0, 1.0, 0.3),),
        degradation="clean")
    b = TrajectoryTrace(
        trace_id="y", split="test", family="heldout", lineage="other",
        generator="g", seed=999, anatomy={"severity": 0.9}, schedule_id="B",
        actions=((1.0, 0.0),), observations=((0.1, 0.2, 0.0, 1.0, 0.3),),
        degradation="clean")
    assert a.content_sha256 == b.content_sha256
    c = TrajectoryTrace(
        trace_id="z", split="train", family="dev", lineage="L", generator="g",
        seed=1, anatomy={"severity": 0.5}, schedule_id="A",
        actions=((1.0, 0.0),), observations=((0.1, 0.2, 0.0, 1.0, 0.300001),),
        degradation="clean")
    assert c.content_sha256 != a.content_sha256


def test_content_hash_has_no_quantization():
    """Any numeric difference, however far below the old rounding grid,
    must change the digest: identical content means EXACT content."""
    base_obs = (0.1, 0.2, 0.0, 1.0, 0.3)
    a = TrajectoryTrace(
        trace_id="a", split="train", family="dev", lineage="L", generator="g",
        seed=1, anatomy={"severity": 0.5}, schedule_id="A",
        actions=((1.0, 0.0),), observations=(base_obs,), degradation="clean")
    nudged = list(base_obs)
    nudged[-1] = base_obs[-1] + 1e-9  # far below the old 6-decimal grid
    d = TrajectoryTrace(
        trace_id="d", split="train", family="dev", lineage="L", generator="g",
        seed=1, anatomy={"severity": 0.5}, schedule_id="A",
        actions=((1.0, 0.0),), observations=(tuple(nudged),),
        degradation="clean")
    assert d.content_sha256 != a.content_sha256


# ---------------------------------------------------------------- audit


def _mk(tid, obs, actions, seed=1, anatomy=None):
    return TrajectoryTrace(
        trace_id=tid, split="train", family="dev", lineage=tid,
        generator="nav", seed=seed, anatomy=anatomy or {"severity": 0.5},
        schedule_id="A", actions=actions, observations=obs, degradation="clean")


ACT = ((1.0, 0.0), (0.8, 0.2))


def test_near_is_caught_regardless_of_metadata():
    obs = ((0.1, 0.2, 0.0, 1.0, 0.3), (0.1, 0.2, 0.0, 1.0, 0.31))
    nudged = tuple(tuple(v + 5e-7 for v in o) for o in obs)
    a = _mk("a", obs, ACT, seed=1)
    b = _mk("b", nudged, ((0.5, 0.5), (0.2, -0.1)), seed=99,
            anatomy={"severity": 0.99})
    result = audit([a, b])
    assert result["counts"]["by_kind"]["near"] == 1
    pair = result["pairs"][0]
    assert pair["max_obs_dev"] == pytest.approx(5e-7)


def test_seed_sibling_with_same_geometry_but_different_actions():
    obs_a = ((0.1, 0.2, 0.0, 1.0, 0.3), (0.1, 0.2, 0.0, 1.0, 0.31))
    obs_b = ((0.6, 0.7, 0.2, 0.4, 0.9), (0.6, 0.7, 0.2, 0.4, 0.91))
    a = _mk("a", obs_a, ACT, seed=7)
    # same anatomy + seed, different actions and distant observations
    b = _mk("b", obs_b, ((0.5, 0.5), (0.2, -0.1)), seed=7)
    result = audit([a, b])
    assert result["counts"]["by_kind"]["seed_sibling"] == 1


def test_declared_parent_siblings_are_linked():
    obs_a = ((0.1, 0.2, 0.0, 1.0, 0.3), (0.1, 0.2, 0.0, 1.0, 0.31))
    obs_b = ((0.9, 0.8, 0.1, 0.2, 0.7), (0.9, 0.8, 0.1, 0.2, 0.71))
    # different anatomy + different seeds so no content-behavior rule applies;
    # only the declared parent provenance links them
    child_a = _mk("ca", obs_a, ACT, seed=1, anatomy={"severity": 0.5})
    child_b = _mk("cb", obs_b, ACT, seed=2, anatomy={"severity": 0.95})
    child_a.parent = child_b.parent = "shared/base"
    result = audit([child_a, child_b])
    assert result["counts"]["by_kind"]["declared"] == 1
    pair = result["pairs"][0]
    assert pair["max_obs_dev"] > 0
    assert pair["shared_frames"] == 0


def test_time_shifted_literal_shared_frames():
    obs4 = ((0.1, 0.2, 0.0, 1.0, 0.3), (0.2, 0.3, 0.1, 0.9, 0.4),
            (0.3, 0.4, 0.2, 0.8, 0.5), (0.4, 0.5, 0.3, 0.7, 0.6))
    shifted = ((0.99, 0.99, 0.99, 0.99, 0.99),) + obs4[:3]
    a = _mk("a", obs4, ACT, seed=1)
    b = _mk("b", shifted, ACT, seed=1)  # b is a time-shifted copy of a
    result = audit([a, b])
    pair = result["pairs"][0]
    assert pair["kind"] == "frame_shared"
    assert pair["shared_frames"] == 3
    assert pair["frame_shift"] != 0
    assert pair["max_obs_dev"] == pytest.approx(0.99)  # elementwise max diff


def test_pairs_carry_distance_and_shared_frame_fields(corpus):
    dev, heldout, _ = corpus
    result = audit(dev + heldout)
    assert result["pairs"], "matched pairs expected"
    for p in result["pairs"]:
        assert "max_obs_dev" in p
        assert "shared_frames" in p
        assert p["max_obs_dev"] is None or p["max_obs_dev"] >= 0
        assert p["shared_frames"] >= 0
    exact = next(p for p in result["pairs"] if p["kind"] == "exact")
    assert exact["max_obs_dev"] == pytest.approx(0.0)
    assert exact["shared_frames"] == len(
        next(t for t in dev + heldout
             if t.trace_id == exact["a"]).observations)


def test_exact_duplicate_across_splits_is_grouped(corpus):
    dev, heldout, _ = corpus
    result = audit(dev + heldout)
    exact = [p for p in result["pairs"] if p["kind"] == "exact"]
    assert exact
    dup_pairs = [p for p in exact
                 if any("#exact" in result["lineage"][tid]["lineage"]
                        for tid in (p["a"], p["b"]))]
    assert dup_pairs
    # dev exact dups deliberately probe the test split (cross-split); the
    # held-out dups stay on the heldout split
    dev_dups = [p for p in dup_pairs
                if result["provenance"][p["a"]]["family"] == "dev"
                or result["provenance"][p["b"]]["family"] == "dev"]
    assert dev_dups and all(p["cross_split"] for p in dev_dups)
    assert any(g["cross_split"] for g in result["groups"])


def test_near_frame_and_perturbation_are_distinguished(corpus):
    dev, heldout, meta = corpus
    result = audit(dev + heldout)
    kinds = result["counts"]["by_kind"]
    assert kinds["near"] >= 1
    assert kinds["frame_shared"] >= 1
    assert kinds["anatomy_action"] >= 1  # shared anatomy+actions, perturbed obs
    # the declared perturbation must NOT be classified as near or frame
    base, variants = next(iter(meta["sibling_variants"].items()))
    pert_id = next(
        t.trace_id for t in dev if t.lineage ==
        next(l for l, d in meta["sibling_variants"][base].items()
             if d == "perturbed"))
    kinds_to_parent = {p["kind"] for p in result["pairs"]
                       if pert_id in (p["a"], p["b"])}
    assert "anatomy_action" in kinds_to_parent
    assert "near" not in kinds_to_parent
    assert "frame_shared" not in kinds_to_parent


def test_seed_sibling_same_generator_seed_different_anatomy():
    a = producer_trace(
        seed=10, split="train", family="dev", lineage="base",
        generator="nav", anatomy={"asset": "stenotic_tube", "severity": 0.5},
        schedule_id="A", action_schedule=((1.0, 0.0), (0.8, 0.2), (1.0, 0.0)))
    b = producer_trace(
        seed=10, split="train", family="dev", lineage="sib",
        generator="nav", anatomy={"asset": "tortuous_tube", "severity": 0.3},
        schedule_id="A", action_schedule=((1.0, 0.0), (0.8, 0.2), (1.0, 0.0)))
    assert a.content_sha256 != b.content_sha256
    result = audit([a, b])
    assert result["counts"]["by_kind"]["seed_sibling"] == 1


def test_transitive_closure_groups_all_related_traces(corpus):
    dev, heldout, meta = corpus
    result = audit(dev + heldout)
    base, variants = next(iter(meta["sibling_variants"].items()))
    ids = [base] + list(variants)
    trace_ids = {t.trace_id for t in dev if t.lineage in ids}
    group = next(g for g in result["groups"]
                 if set(g["trajectories"]) & trace_ids)
    assert trace_ids <= set(group["trajectories"])
    linked = {tid for tid in trace_ids
              if result["lineage"][tid]["links"]}
    assert linked == trace_ids


def test_audit_has_no_bogus_negative_control_default(corpus):
    dev, heldout, _ = corpus
    result = audit(dev + heldout)
    assert "negative_control" not in result


def test_baseline_has_no_inflated_evidence_grade(corpus):
    dev, _, _ = corpus
    train = [t for t in dev if t.split == "train" and t.degradation == "clean"]
    test = [t for t in dev if t.split == "test"]
    result = run_baseline(train, test)
    assert "evidence_grade" not in result


# ------------------------------------------------------- frozen holdout


def test_frozen_holdout_nonempty_stable_and_excludes_development(corpus):
    dev, heldout, meta = corpus
    result = audit(dev + heldout)
    held = result["heldout"]
    # independence is VERIFIED against every full transitive group
    assert held["development_nonempty"] is True
    assert held["crossing_groups"] == []
    assert held["independent"] is True
    assert held["complete"] is True
    assert held["groups"], "frozen holdout must be nonempty"
    held_lineages = {lg for g in held["groups"] for lg in g["lineages"]}
    assert set(meta["holdout_lineages"]) <= held_lineages
    for g in held["groups"]:
        for tid in g["trajectories"]:
            assert result["provenance"][tid]["family"] == "heldout"
    r2 = audit(dev + heldout)
    assert held["frozen_digest"] == r2["heldout"]["frozen_digest"]
    # membership change invalidates the digest
    dev3, held3, _ = build_corpus()
    extra = producer_trace(
        seed=77, split="heldout", family="heldout", lineage="heldout/extra",
        generator="nav", anatomy={"asset": "tortuous_tube", "severity": 0.45},
        schedule_id="E",
        action_schedule=((1.0, 0.0), (0.5, 0.1), (1.0, 0.0), (0.2, 0.0)))
    r3 = audit(dev3 + held3 + [extra])
    assert r3["heldout"]["frozen_digest"] != held["frozen_digest"]
    assert r3["heldout"]["independent"] is True


def test_cross_family_content_plant_breaks_frozen_independence(corpus):
    """A held-out trace's content planted into the development family must
    be DETECTED: the transitive group crosses families, so the frozen
    holdout is reported not independent (no retrospective filtering)."""
    dev, heldout, _ = corpus
    src = next(t for t in heldout if t.degradation == "clean")
    planted = dataclasses.replace(
        src, trace_id="planted/dev-copy", lineage="planted",
        family="dev", split="train", parent=None,
        provenance={"source": "planted_cross_family"})
    result = audit(dev + heldout + [planted])
    held = result["heldout"]
    exact_cross = [p for p in result["pairs"] if p["kind"] == "exact"
                   and "planted/dev-copy" in (p["a"], p["b"])]
    assert exact_cross, "cross-family content copy must be detected as exact"
    assert held["independent"] is False
    assert held["crossing_groups"], "crossing groups must be reported"
    # no retrospective filtering: the leaked content stays in the groups
    all_grouped = {tid for g in result["groups"]
                   for tid in g["trajectories"]}
    assert {"planted/dev-copy", src.trace_id} <= all_grouped


def test_holdout_metrics_suppressed_when_frozen_audit_fails(corpus):
    dev, heldout, _ = corpus
    train = [t for t in dev if t.split == "train"
             and t.degradation == "clean"]
    bad = {"independent": False, "groups": []}
    out = evaluate_heldout(train, heldout, bad)
    assert out["suppressed"] is True
    assert "traces" not in out
    # empty holdout is equally suppressed
    out2 = evaluate_heldout(train, heldout, {"independent": True, "groups": []})
    assert out2["suppressed"] is True


# ---------------------------------------------------- negative control


def _independent_traces(corpus):
    dev, heldout, _ = corpus
    return [t for t in dev if t.degradation == "clean"] + [
        t for t in heldout if t.degradation == "clean"]


def test_negative_control_detects_injected_leak(corpus):
    result = audit_with_negative_control(_independent_traces(corpus), inject=True)
    nc = result["negative_control"]
    assert nc["injected"]
    assert set(nc["detected"]) == set(nc["injected"])
    assert nc["passed"] is True
    assert result["counts"]["by_kind"]["exact"] >= 1
    # the leak must never land in the frozen holdout groups
    leaked_groups = {result["lineage"][tid]["group_id"] for tid in nc["detected"]}
    assert not (leaked_groups & {g["group_id"] for g in result["heldout"]["groups"]})


def test_negative_control_clean_run_has_no_detections(corpus):
    result = audit_with_negative_control(_independent_traces(corpus), inject=False)
    nc = result["negative_control"]
    assert nc["injected"] == [] and nc["detected"] == []
    assert nc["passed"] is True


def test_negative_control_fails_when_detector_broken(corpus, monkeypatch):
    import lumen.experiments.lineage_audit as la
    monkeypatch.setattr(la, "_pair_kind", lambda *a, **k: None)
    result = audit_with_negative_control(_independent_traces(corpus), inject=True)
    assert result["negative_control"]["passed"] is False


# ------------------------------------------------------------- producer


def test_producer_is_actual_navenv_and_deterministic():
    kwargs = dict(
        split="train", family="dev", lineage="d", generator="nav",
        anatomy={"asset": "stenotic_tube", "severity": 0.5}, schedule_id="A",
        action_schedule=((1.0, 0.0), (0.8, 0.2), (1.0, 0.0), (0.5, 0.0)))
    a1 = producer_trace(seed=0, **kwargs)
    a2 = producer_trace(seed=0, **kwargs)
    b = producer_trace(seed=1, **kwargs)
    assert a1.content_sha256 == a2.content_sha256
    assert a1.content_sha256 != b.content_sha256
    assert a1.observations and len(a1.observations) == len(a1.actions)
    obs0 = np.asarray(a1.observations[0])
    assert obs0.shape == (5,)          # 5-state observations
    assert np.isfinite(obs0).all()
    assert a1.actions[0] == (1.0, 0.0)  # fixed 2-action schedule


# -------------------------------------------------------------- baseline


def test_baseline_matches_original_fixed_schema(corpus):
    dev, heldout, _ = corpus
    train = [t for t in dev if t.split == "train" and t.degradation == "clean"]
    test = [t for t in dev if t.split == "test"]
    result = run_baseline(train, test)
    assert result["schema"] == "seldinger-world-model-baseline-v1"
    assert result["train_transitions"] == sum(len(t.actions) - 1 for t in train)
    assert result["test_transitions"] == sum(len(t.actions) - 1 for t in test)
    assert np.isfinite(result["persistence_mse"])
    assert np.isfinite(result["action_conditioned_mse"])
    assert result["decision"] in ("continue",
                                  "negative result; change model before follow-on")
    assert result["calibration"]["warning"].startswith("retrospective")
    json.dumps(result)


def test_baseline_runs_separately_on_heldout_family(corpus):
    dev, heldout, _ = corpus
    train = [t for t in dev if t.split == "train" and t.degradation == "clean"]
    result = run_baseline(train, heldout)
    assert result["test_transitions"] == sum(len(t.actions) - 1 for t in heldout)
    json.dumps(result)


# --------------------------------------------------- replacement entry


def test_replacement_audit_returns_json_dict_with_stratified_holdout(corpus):
    result = replacement_audit()
    json.dumps(result)
    assert result["schema"].startswith("lumen-lineage-audit/")
    assert "corpus" in result and "baseline" in result
    assert result["provenance"]["data"] == "synthetic procedural NavEnv rollouts"
    for stratum in STRATA:
        if stratum == "declared":
            continue  # corpus has no cross-seed declared siblings; unit-tested
        assert result["audit"]["degradation"]["counts"][stratum] > 0
    assert result["baseline"]["dev"]["schema"] == \
        "seldinger-world-model-baseline-v1"
    assert result["negative_control"]["passed"] is True
    # per-heldout-trace metrics, stratified by actual perturbation
    held = result["baseline"]["heldout"]
    assert held["suppressed"] is False
    assert len(held["traces"]) == 10
    by_deg = {}
    for tid, m in held["traces"].items():
        assert m["schema"] == "seldinger-world-model-baseline-v1"
        assert np.isfinite(m["action_conditioned_mse"])
        by_deg.setdefault(m["degradation"], []).append(tid)
    assert set(by_deg) == {"clean", "near", "frame", "perturbed", "exact"}
    # primary training kept clean-dev only: no planted dev copies
    assert result["baseline"]["dev"]["train_transitions"] == sum(
        len(t.actions) - 1 for t in corpus[0]
        if t.split == "train" and t.degradation == "clean")


# ---------------------------------------------------------------- hygiene


def test_duplicate_trace_ids_rejected(corpus):
    dev, _, _ = corpus
    with pytest.raises(ValueError):
        audit(dev + [dev[0]])


def test_leaky_control_requires_two_splits(corpus):
    dev, _, _ = corpus
    same_split = [t for t in dev if t.split == "train"]
    with pytest.raises(ValueError):
        leaky_control(same_split)


def test_clean_control_rejects_an_existing_cross_split_duplicate(corpus):
    traces, _ = leaky_control(_independent_traces(corpus))
    result = audit_with_negative_control(traces, inject=False)
    assert not result["negative_control"]["passed"]
