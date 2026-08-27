import json

import pytest

from lumen.bench_protocol import (
    GeneralizationReport,
    ScaledSuite,
    SplitTask,
    evaluate_generalization,
    make_scaled_suite,
    validate_report,
)


def test_scaled_suite_has_frozen_disjoint_case_manifest():
    suite = make_scaled_suite()

    assert suite.version == "lumen-bench/heldout/1"
    assert [task.episodes for task in suite.train] == [100, 100, 100]
    assert [task.episodes for task in suite.heldout] == [100, 100, 100]
    assert [task.seed for task in suite.train] == [10_000, 11_000, 12_000]
    assert [task.seed for task in suite.heldout] == [20_000, 21_000, 22_000]
    train_ids = {task.case_id for task in suite.train}
    heldout_ids = {task.case_id for task in suite.heldout}
    assert train_ids.isdisjoint(heldout_ids)
    assert len(suite.manifest()) == 6
    assert {item["split"] for item in suite.manifest()} == {"train", "heldout"}


def test_scaled_suite_rejects_invalid_episode_count_and_split_container():
    with pytest.raises(ValueError, match="positive integer"):
        make_scaled_suite(0)
    with pytest.raises(ValueError, match="positive integer"):
        make_scaled_suite(1.5)

    task = SplitTask("case", "heldout", "case-id", "easy", lambda: None, 1, 0)
    with pytest.raises(ValueError, match="train tasks must use the 'train' split"):
        ScaledSuite(train=(task,), heldout=(
            SplitTask("other", "heldout", "other-id", "easy", lambda: None, 1, 1),
        ))
    with pytest.raises(ValueError, match="non-empty train and heldout"):
        ScaledSuite(train=(), heldout=(task,))


def _fake_task_result(task):
    train = task.name.startswith("train_")
    success = 0.8 if train else 0.4
    return {
        "name": task.name,
        "tier": task.tier,
        "episodes": task.episodes,
        "success_rate": success,
        "safe_success_rate": success - 0.1,
        "unsafe_success_rate": 0.1,
        "crash_rate": 0.0,
        "mean_return": 10.0 if train else 4.0,
        "max_pen": 0.2 if train else 0.4,
        "_episode_metrics": {
            "success_rate": [success] * task.episodes,
            "safe_success_rate": [success - 0.1] * task.episodes,
            "unsafe_success_rate": [0.1] * task.episodes,
            "crash_rate": [0.0] * task.episodes,
            "mean_return": [10.0 if train else 4.0] * task.episodes,
            "max_pen": [0.2 if train else 0.4] * task.episodes,
        },
    }


def test_evaluate_generalization_reports_split_means_gaps_and_round_trips(tmp_path, monkeypatch):
    import lumen.bench_protocol as protocol

    monkeypatch.setattr(protocol, "evaluate_task", lambda task, _policy: _fake_task_result(task))
    report = evaluate_generalization(lambda _obs: 0, name="fake", episodes_per_task=3)
    assert report.statistics["train"]["metrics"]["success_rate"]["iqm"] == pytest.approx(0.8)
    assert report.statistics["heldout"]["metrics"]["success_rate"]["n"] == 9

    assert report.name == "fake"
    assert report.train["success_rate"] == pytest.approx(0.8)
    assert report.heldout["success_rate"] == pytest.approx(0.4)
    assert report.generalization_gap["success_rate"] == pytest.approx(0.4)
    assert report.generalization_gap["mean_return"] == pytest.approx(6.0)
    assert all(task["episodes"] == 3 for task in report.train["tasks"] + report.heldout["tasks"])

    path = tmp_path / "generalization.json"
    report.save(path)
    loaded = GeneralizationReport.load(path)
    assert loaded.to_dict() == json.loads(path.read_text())
    assert validate_report(loaded) is loaded


def test_validate_report_rejects_split_label_mismatch():
    suite = make_scaled_suite(episodes_per_task=1)
    report = GeneralizationReport(
        name="bad",
        suite_version=suite.version,
        manifest=suite.manifest(),
        train={
            "tasks": [{
                "split": "heldout",
                "success_rate": 0.0,
                "safe_success_rate": 0.0,
                "unsafe_success_rate": 0.0,
                "crash_rate": 0.0,
                "mean_return": 0.0,
                "max_pen": 0.0,
            }],
            "success_rate": 0.0,
            "safe_success_rate": 0.0,
            "unsafe_success_rate": 0.0,
            "crash_rate": 0.0,
            "mean_return": 0.0,
            "max_pen": 0.0,
        },
        heldout={
            "tasks": [{
                "split": "heldout",
                "success_rate": 0.0,
                "safe_success_rate": 0.0,
                "unsafe_success_rate": 0.0,
                "crash_rate": 0.0,
                "mean_return": 0.0,
                "max_pen": 0.0,
            }],
            "success_rate": 0.0,
            "safe_success_rate": 0.0,
            "unsafe_success_rate": 0.0,
            "crash_rate": 0.0,
            "mean_return": 0.0,
            "max_pen": 0.0,
        },
        generalization_gap={metric: 0.0 for metric in (
            "success_rate", "safe_success_rate", "unsafe_success_rate", "crash_rate",
            "mean_return", "max_pen",
        )},
    )
    with pytest.raises(ValueError, match="split labels"):
        validate_report(report)


def test_benchmark_cli_scaled_writes_gap_report(tmp_path, monkeypatch, capsys):
    import lumen.bench_protocol as protocol
    from lumen.cli import benchmark_main

    class DummyReport:
        suite_version = "lumen-bench/heldout/1"
        name = "forward-baseline"
        train = {"success_rate": 0.8, "safe_success_rate": 0.7}
        heldout = {"success_rate": 0.4, "safe_success_rate": 0.3}
        generalization_gap = {"success_rate": 0.4}

        def save(self, path):
            with open(path, "w") as file:
                json.dump({"suite": self.suite_version}, file)

    seen = {}

    def fake_evaluate(policy, name, episodes_per_task):
        seen.update(policy=policy, name=name, episodes=episodes_per_task)
        return DummyReport()

    monkeypatch.setattr(protocol, "evaluate_generalization", fake_evaluate)
    benchmark_main([str(tmp_path), "--suite", "scaled", "--episodes", "2"])
    output = json.loads(capsys.readouterr().out)

    assert seen["name"] == "forward-baseline"
    assert seen["episodes"] == 2
    assert callable(seen["policy"])
    assert output["heldout_success_rate"] == 0.4
    assert output["generalization_gap"] == 0.4
    assert (tmp_path / "forward-baseline-heldout.json").exists()
