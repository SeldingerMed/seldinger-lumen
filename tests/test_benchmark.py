"""M5 benchmark + leaderboard: a fixed suite, a scorecard, and a ranking. The evaluation
drives the Newton sim, so it pytest.importorskips warp/newton (the scorecard/leaderboard
plumbing is pure)."""

import numpy as np
import pytest

from lumen.bench import (SUITE, SUITE_VERSION, Scorecard, evaluate_policy, forward_policy,
                         leaderboard, run_episode, scorecard_rejections, validate_scorecard)


def test_suite_is_fixed_and_tiered():
    names = [t.name for t in SUITE]
    assert names == ["nav_tube", "nav_stenotic", "nav_tree_branch"]   # canonical, ordered
    assert [t.tier for t in SUITE] == ["easy", "medium", "hard"]


def test_forward_baseline_scores_the_whole_suite():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    sc = evaluate_policy(forward_policy, "forward-baseline")
    assert sc.suite_version == SUITE_VERSION and len(sc.per_task) == 3
    assert sc.overall["success_rate"] == 1.0          # the baseline solves the suite...
    assert 0.0 <= sc.overall["safe_success_rate"] < sc.overall["success_rate"]
    assert sc.per_task[2]["mean_steps"] > sc.per_task[0]["mean_steps"]   # ...the tree costs more steps
    assert all(np.isfinite([t["safe_success_rate"], t["max_pen"], t["mean_return"]]).all()
               for t in sc.per_task)


def test_a_better_policy_outranks_the_baseline_on_the_leaderboard(tmp_path):
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    # the baseline + a deliberately worse policy (does nothing -> never reaches the target)
    base = evaluate_policy(forward_policy, "forward-baseline")
    stuck = evaluate_policy(lambda obs: np.array([0.0], np.float32), "do-nothing")
    base.save(tmp_path / "a_base.json")
    stuck.save(tmp_path / "b_stuck.json")
    board = leaderboard(str(tmp_path))
    assert [c.name for c in board] == ["forward-baseline", "do-nothing"]   # higher success first
    assert stuck.overall["success_rate"] == 0.0


def test_scorecard_round_trips_and_skips_foreign_suite_versions(tmp_path):
    sc = Scorecard(name="x", suite_version=SUITE_VERSION,
                   per_task=[
                       {"name": t.name, "tier": t.tier, "episodes": t.episodes,
                        "success_rate": 1.0, "safe_success_rate": 1.0,
                        "max_pen": 0.0, "mean_return": 1.0}
                       for t in SUITE
                   ],
                   overall={"success_rate": 1.0, "safe_success_rate": 1.0,
                            "max_pen": 0.0, "mean_return": 1.0})
    sc.save(tmp_path / "x.json")
    assert Scorecard.load(tmp_path / "x.json").overall == sc.overall      # round-trip
    # a scorecard from a different suite version is not comparable -> excluded from the board
    Scorecard(name="old", suite_version="lumen-bench/HUH", per_task=[],
              overall={"success_rate": 1.0, "max_pen": 0.0, "mean_return": 0.0}).save(tmp_path / "old.json")
    assert [c.name for c in leaderboard(str(tmp_path))] == ["x"]


def test_scorecard_validation_reports_submission_schema_errors():
    missing_safe = Scorecard(name="bad", suite_version=SUITE_VERSION,
                             per_task=[{"name": "nav_tube", "success_rate": 1.0,
                                        "max_pen": 0.0}],
                             overall={"success_rate": 1.0, "max_pen": 0.0,
                                      "mean_return": 0.0})
    with pytest.raises(ValueError, match="safe_success_rate"):
        validate_scorecard(missing_safe)

    missing_task = Scorecard(name="partial", suite_version=SUITE_VERSION,
                             per_task=[{"name": "nav_tube", "success_rate": 1.0,
                                        "safe_success_rate": 1.0, "max_pen": 0.0,
                                        "mean_return": 0.0}],
                             overall={"success_rate": 1.0, "safe_success_rate": 1.0,
                                      "max_pen": 0.0, "mean_return": 0.0})
    with pytest.raises(ValueError, match="per_task names"):
        validate_scorecard(missing_task)

    wrong_tier = Scorecard(name="wrong-tier", suite_version=SUITE_VERSION,
                           per_task=[
                               {"name": t.name, "tier": "easy", "episodes": t.episodes,
                                "success_rate": 1.0, "safe_success_rate": 1.0,
                                "max_pen": 0.0, "mean_return": 1.0}
                               for t in SUITE
                           ],
                           overall={"success_rate": 1.0, "safe_success_rate": 1.0,
                                    "max_pen": 0.0, "mean_return": 1.0})
    with pytest.raises(ValueError, match="per_task\\[2\\].tier"):
        validate_scorecard(wrong_tier)

    inflated_overall = Scorecard(name="inflated", suite_version=SUITE_VERSION,
                                 per_task=[
                                     {"name": t.name, "tier": t.tier, "episodes": t.episodes,
                                      "success_rate": 1.0, "safe_success_rate": 0.0,
                                      "max_pen": 0.0, "mean_return": 1.0}
                                     for t in SUITE
                                 ],
                                 overall={"success_rate": 1.0, "safe_success_rate": 1.0,
                                          "max_pen": 0.0, "mean_return": 1.0})
    with pytest.raises(ValueError, match="overall.safe_success_rate"):
        validate_scorecard(inflated_overall)

    patient_card = Scorecard(name="private", suite_version=SUITE_VERSION,
                             per_task=[
                                 {"name": t.name, "tier": t.tier, "episodes": t.episodes,
                                  "success_rate": 1.0, "safe_success_rate": 1.0,
                                  "max_pen": 0.0, "mean_return": 1.0}
                                 for t in SUITE
                             ],
                             overall={"success_rate": 1.0, "safe_success_rate": 1.0,
                                      "max_pen": 0.0, "mean_return": 1.0},
                             provenance="patient(private)")
    with pytest.raises(ValueError, match="provenance"):
        validate_scorecard(patient_card)


def test_scorecard_rejections_explain_why_submissions_are_skipped(tmp_path):
    bad = Scorecard(name="bad", suite_version=SUITE_VERSION,
                    per_task=[{"name": "nav_tube", "success_rate": 1.0,
                               "max_pen": 0.0}],
                    overall={"success_rate": 1.0, "max_pen": 0.0,
                             "mean_return": 0.0})
    bad.save(tmp_path / "bad.json")

    rejected = scorecard_rejections(str(tmp_path))

    assert len(rejected) == 1
    assert rejected[0]["path"].endswith("bad.json")
    assert "safe_success_rate" in rejected[0]["error"]


def test_run_episode_reports_finite_metrics():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    env = SUITE[0].make_env()
    out = run_episode(env, forward_policy, seed=0)
    assert set(out) == {"success", "safe_success", "steps", "max_pen", "return", "clinical"}
    assert out["success"] and out["steps"] > 0 and np.isfinite(out["return"])
    assert out["clinical"]["tip_target"]["success"] is True
    assert out["clinical"]["wall_safety"]["max_penetration"] >= 0.0


def test_leaderboard_ranks_clinically_safe_success_before_unsafe_target_hits(tmp_path):
    unsafe = Scorecard(name="unsafe-fast", suite_version=SUITE_VERSION, per_task=[
                           {"name": t.name, "tier": t.tier, "episodes": t.episodes,
                            "success_rate": 1.0, "safe_success_rate": 0.0,
                            "max_pen": 2.0, "mean_return": 100.0}
                           for t in SUITE
                       ],
                       overall={"success_rate": 1.0, "safe_success_rate": 0.0,
                                "max_pen": 2.0, "mean_return": 100.0})
    safe = Scorecard(name="safe-partial", suite_version=SUITE_VERSION, per_task=[
                         {"name": t.name, "tier": t.tier, "episodes": t.episodes,
                          "success_rate": 0.8, "safe_success_rate": 0.8,
                          "max_pen": 0.0, "mean_return": 20.0}
                         for t in SUITE
                     ],
                     overall={"success_rate": 0.8, "safe_success_rate": 0.8,
                              "max_pen": 0.0, "mean_return": 20.0})
    unsafe.save(tmp_path / "unsafe.json")
    safe.save(tmp_path / "safe.json")

    assert [c.name for c in leaderboard(str(tmp_path))] == ["safe-partial", "unsafe-fast"]


def test_submit_policy_example_writes_a_comparable_scorecard(tmp_path):
    pytest.importorskip("warp")
    pytest.importorskip("newton")

    from examples.submit_policy import main

    out = main(str(tmp_path), name="example-policy")
    card = validate_scorecard(Scorecard.load(out))

    assert card.name == "example-policy"
    assert [c.name for c in leaderboard(str(tmp_path))] == ["example-policy"]
