import copy

import numpy as np

from lumen.bench import (
    BenchTask,
    Scorecard,
    evaluate_policy,
    replay_verified_leaderboard,
    validate_scorecard,
    verify_replay,
)


class OneStepEnv:
    R = 2.0
    target_s = 1.0
    success_tol = 2.5
    safety_max_pen = 0.3

    def reset(self, *, seed=None):
        self.seed = seed
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        self.action = np.asarray(action).copy()
        return (
            np.zeros(5, dtype=np.float32),
            1.0,
            True,
            False,
            {"dist": 0.0, "max_pen": 0.0, "success": True, "diverged": False},
        )


def _suite():
    return [BenchTask("replay_case", "easy", OneStepEnv, episodes=2, seed=7)]


def test_evaluation_emits_replay_certificate_that_replays():
    policy = lambda _obs: np.array([0.25, 0.0], dtype=np.float32)
    card = evaluate_policy(policy, "replay-policy", suite=_suite())

    assert card.replay["protocol"] == "lumen-bench-replay/1"
    assert len(card.replay["episodes"]) == 2
    assert all(len(row["sha256"]) == 64 for row in card.replay["episodes"])
    validate_scorecard(card, suite=_suite())
    report = verify_replay(card, policy, suite=_suite())
    assert report == {
        "protocol": "lumen-bench-replay/1",
        "verified": True,
        "episodes": 2,
        "errors": [],
    }


def test_replay_verification_rejects_tampering_and_changed_policy():
    policy = lambda _obs: np.array([0.25, 0.0], dtype=np.float32)
    card = evaluate_policy(policy, "replay-policy", suite=_suite())

    tampered = copy.deepcopy(card)
    tampered.replay["episodes"][0]["actions"][0][0] = 0.75
    report = verify_replay(tampered, policy, suite=_suite())
    assert not report["verified"]
    assert any("sha256" in error for error in report["errors"])

    changed = verify_replay(
        card,
        lambda _obs: np.array([-0.25, 0.0], dtype=np.float32),
        suite=_suite(),
    )
    assert not changed["verified"]
    assert any("digest mismatch" in error for error in changed["errors"])



def test_replay_verification_rejects_aggregate_metric_tampering():
    policy = lambda _obs: np.array([0.25, 0.0], dtype=np.float32)
    card = evaluate_policy(policy, "replay-policy", suite=_suite())
    tampered = copy.deepcopy(card)
    tampered.per_task[0]["success_rate"] = 0.0
    tampered.overall["success_rate"] = 0.0

    report = verify_replay(tampered, policy, suite=_suite())

    assert not report["verified"]
    assert any("replay overall.success_rate" in error for error in report["errors"])

def test_replay_verified_leaderboard_requires_supplied_policy(tmp_path):
    policy = lambda _obs: np.array([0.25, 0.0], dtype=np.float32)
    card = evaluate_policy(policy, "replay-policy", suite=_suite())
    card.save(tmp_path / "replay.json")

    assert [c.name for c in replay_verified_leaderboard(
        str(tmp_path), {"replay-policy": policy}, suite=_suite()
    )] == ["replay-policy"]
    assert replay_verified_leaderboard(str(tmp_path), {}, suite=_suite()) == []


def test_scorecard_loads_legacy_json_as_unverified(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"name":"legacy","suite_version":"lumen-bench/3",'
        '"per_task":[],"overall":{}}'
    )
    card = Scorecard.load(path)
    assert card.replay == {}
    assert not verify_replay(card, lambda _obs: [0.0, 0.0], suite=[])["verified"]
