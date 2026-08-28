import json
import sys
from pathlib import Path

import pytest
pytest.importorskip("gymnasium")


def test_parse_seed_schedule_requires_unique_nonnegative_integers():
    from benchmarks.external_comparison.train_sb3 import parse_seed_schedule

    assert parse_seed_schedule("0, 7, 12") == [0, 7, 12]
    with pytest.raises(ValueError, match="unique"):
        parse_seed_schedule("1,1")
    with pytest.raises(ValueError, match="non-negative"):
        parse_seed_schedule("-1")


def test_run_id_is_sanitized_to_one_filename_component():
    from benchmarks.external_comparison.train_sb3 import sanitize_run_id

    assert sanitize_run_id("../multi seed") == "multi-seed"
    with pytest.raises(ValueError, match="safe filename"):
        sanitize_run_id("..")


def test_main_trains_and_evaluates_each_requested_seed(tmp_path, monkeypatch, capsys):
    from benchmarks.external_comparison import train_sb3

    class FakeEnv:
        def close(self):
            pass

    class FakeModel:
        def __init__(self, _policy, _env, **kwargs):
            self.seed = kwargs["seed"]

        def learn(self, total_timesteps, progress_bar):
            assert total_timesteps == 8
            assert progress_bar is False

        def save(self, path):
            path.write_text(str(self.seed))

    writes = {}

    def fake_evaluate(args, model, *, training_seed, model_id):
        assert args.seed == model.seed
        return [
            train_sb3.EpisodeResult(
                environment=args.environment,
                task=args.task,
                task_class="simple_target_navigation",
                policy="ppo_trained",
                seed=train_sb3.EVAL_SEED_START + ep_idx,
                training_seed=training_seed,
                model_id=model_id,
                success=True,
                steps=2,
                total_reward=3.0,
                final_distance=0.0,
            )
            for ep_idx in range(args.eval_episodes)
        ]

    def fake_write(out_dir, run_id, environment, task_specs, episodes, extra):
        writes.update(
            out_dir=out_dir,
            run_id=run_id,
            environment=environment,
            task_specs=task_specs,
            episodes=episodes,
            extra=extra,
        )

    monkeypatch.setattr(train_sb3, "ALGOS", {"ppo": FakeModel, "sac": FakeModel})
    monkeypatch.setattr(train_sb3, "make_env", lambda *_args: FakeEnv())
    monkeypatch.setattr(train_sb3, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(train_sb3, "_write_results", fake_write)
    monkeypatch.setattr(train_sb3, "_aggregate", lambda episodes: {"episodes": len(episodes)})
    monkeypatch.setattr(train_sb3, "_host_snapshot", lambda: {})
    monkeypatch.setattr(train_sb3, "_git_commit", lambda _path: "commit")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sb3.py",
            "--environment", "lumen",
            "--task", "nav_tube",
            "--algo", "ppo",
            "--timesteps", "8",
            "--eval-episodes", "4",
            "--seeds", "11,22",
            "--run-id", "multi",
            "--out-dir", str(tmp_path / "results"),
            "--model-dir", str(tmp_path / "models"),
            "--verbose", "0",
        ],
    )

    train_sb3.main()
    output = json.loads(capsys.readouterr().out)

    assert output["seeds"] == [11, 22]
    assert [run["seed"] for run in output["runs"]] == [11, 22]
    assert output["models"] == [
        str(tmp_path / "models" / "multi-seed11.zip"),
        str(tmp_path / "models" / "multi-seed22.zip"),
    ]
    assert {episode.training_seed for episode in writes["episodes"]} == {11, 22}
    assert {episode.seed for episode in writes["episodes"]} == {10_000, 10_001, 10_002, 10_003}
    assert {
        episode.model_id for episode in writes["episodes"]
    } == {
        str(tmp_path / "models" / "multi-seed11.zip"),
        str(tmp_path / "models" / "multi-seed22.zip"),
    }
    assert writes["extra"]["evaluation_seed_policy"] == "common_frozen"
    assert len(writes["episodes"]) == 8
    assert writes["extra"]["seed_count"] == 2
    assert writes["extra"]["seeds"] == [11, 22]
    assert writes["extra"]["eval_episodes_per_seed"] == 4
    assert writes["extra"]["preregistered_main_schedule"] is False
    assert writes["extra"]["runs"] == output["runs"]


def test_aggregate_keeps_training_runs_separate():
    from benchmarks.external_comparison.common_bench import EpisodeResult, _aggregate

    def result(training_seed, model_id, eval_seed):
        return EpisodeResult(
            environment="lumen",
            task="nav_tube",
            task_class="simple_target_navigation",
            policy="ppo_trained",
            seed=eval_seed,
            training_seed=training_seed,
            model_id=model_id,
            success=True,
            steps=2,
            total_reward=1.0,
            final_distance=0.0,
        )

    rows = _aggregate([result(10, "model-10", 10_000), result(2, "model-2", 10_000)])

    assert len(rows) == 2
    assert [row["training_seed"] for row in rows] == [2, 10]
    assert {(row["training_seed"], row["model_id"]) for row in rows} == {
        (2, "model-2"), (10, "model-10")
    }


def test_evaluation_seed_schedule_is_common_and_training_attribution_is_preserved(monkeypatch):
    from types import SimpleNamespace

    from benchmarks.external_comparison import train_sb3

    class FakeEnv:
        def reset(self, *, seed=None):
            return 0, {}

        def step(self, _action):
            return 0, 1.0, True, False, {"success": True, "max_pen": 0.0, "dist": 0.0}

        def close(self):
            pass

    class FakeModel:
        def predict(self, _obs, deterministic):
            assert deterministic is True
            return 0, None

    args = SimpleNamespace(
        eval_episodes=3,
        seed=99,
        environment="lumen",
        task="nav_tube",
        max_steps=4,
        algo="ppo",
        timesteps=8,
    )
    monkeypatch.setattr(train_sb3, "make_env", lambda *_args: FakeEnv())

    first = train_sb3.evaluate_model(args, FakeModel(), training_seed=1, model_id="model-1")
    second = train_sb3.evaluate_model(args, FakeModel(), training_seed=2, model_id="model-2")

    assert [episode.seed for episode in first] == [10_000, 10_001, 10_002]
    assert [episode.seed for episode in second] == [10_000, 10_001, 10_002]
    assert {episode.training_seed for episode in first} == {1}
    assert {episode.model_id for episode in second} == {"model-2"}


def test_common_result_writer_sanitizes_run_id(tmp_path, capsys):
    from benchmarks.external_comparison.common_bench import _write_results

    _write_results(tmp_path, "../unsafe run", "lumen", [], [])
    capsys.readouterr()

    assert (tmp_path / "unsafe-run.json").exists()
    assert (tmp_path / "unsafe-run.csv").exists()


def test_main_reports_training_seed_failures_without_dropping_the_run(
    tmp_path, monkeypatch, capsys
):
    from benchmarks.external_comparison import train_sb3

    writes = {}

    def fake_train(args, seed, model_path):
        if seed == 2:
            raise train_sb3.SeedRunError(
                "evaluate", RuntimeError("synthetic evaluation failure"), 4.5, 1.5
            )
        return model_path, 1.5, 3.0, []

    def fake_write(out_dir, run_id, environment, task_specs, episodes, extra):
        writes.update(episodes=episodes, extra=extra)

    monkeypatch.setattr(train_sb3, "ALGOS", {"ppo": object(), "sac": object()})
    monkeypatch.setattr(train_sb3, "_train_one", fake_train)
    monkeypatch.setattr(train_sb3, "_write_results", fake_write)
    monkeypatch.setattr(train_sb3, "_aggregate", lambda episodes: {"episodes": len(episodes)})
    monkeypatch.setattr(train_sb3, "_host_snapshot", lambda: {})
    monkeypatch.setattr(train_sb3, "_git_commit", lambda _path: "commit")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sb3.py",
            "--environment", "lumen",
            "--task", "nav_tube",
            "--algo", "ppo",
            "--timesteps", "8",
            "--eval-episodes", "2",
            "--seeds", "1,2",
            "--run-id", "failure-run",
            "--out-dir", str(tmp_path / "results"),
            "--model-dir", str(tmp_path / "models"),
        ],
    )

    train_sb3.main()
    output = json.loads(capsys.readouterr().out)

    assert len(writes["episodes"]) == 2
    assert {episode.training_seed for episode in writes["episodes"]} == {2}
    assert all(episode.crashed for episode in writes["episodes"])
    assert writes["episodes"][0].notes["failure_stage"] == "evaluate"
    assert writes["episodes"][0].notes["failure_reason"] == "evaluate_failed"
    assert writes["episodes"][0].notes["failure_elapsed_sec"] == 4.5
    assert writes["extra"]["failure_stages"] == {"2": "evaluate"}
    assert writes["extra"]["failed_seeds"] == [2]
    assert writes["extra"]["successful_seed_count"] == 1
    assert [run["elapsed_sec"] for run in output["runs"]] == [3.0, 4.5]


def test_smoke_steve_sanitizes_result_path(tmp_path, capsys):
    from types import SimpleNamespace

    from benchmarks.external_comparison.common_bench import smoke_steve

    smoke_steve(SimpleNamespace(out_dir=str(tmp_path), run_id="../steve smoke"))
    payload = json.loads(capsys.readouterr().out)

    assert payload["run_id"] == "steve-smoke"
    assert (tmp_path / "steve-smoke.json").exists()


def test_main_rejects_unknown_task_before_training(monkeypatch, capsys):
    from benchmarks.external_comparison import train_sb3

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_sb3.py", "--environment", "lumen", "--task", "missing", "--algo", "ppo"],
    )

    with pytest.raises(SystemExit) as exc_info:
        train_sb3.main()

    assert exc_info.value.code == 2
    assert "--task must be one of" in capsys.readouterr().err


@pytest.mark.parametrize("failure_mode", ["init", "reset"])
def test_evaluation_setup_failures_stay_per_episode(monkeypatch, failure_mode):
    from types import SimpleNamespace

    from benchmarks.external_comparison import train_sb3

    calls = 0

    class FakeEnv:
        def reset(self, *, seed=None):
            if failure_mode == "reset" and seed == train_sb3.EVAL_SEED_START:
                raise RuntimeError("synthetic reset failure")
            return 0, {}

        def step(self, _action):
            return 0, 1.0, True, False, {"success": True, "max_pen": 0.0, "dist": 0.0}

        def close(self):
            pass

    class FakeModel:
        def predict(self, _obs, deterministic):
            assert deterministic is True
            return 0, None

    def make_fake_env(*_args):
        nonlocal calls
        calls += 1
        if failure_mode == "init" and calls == 1:
            raise RuntimeError("synthetic env init failure")
        return FakeEnv()

    args = SimpleNamespace(
        eval_episodes=2,
        seed=99,
        environment="lumen",
        task="nav_tube",
        max_steps=4,
        algo="ppo",
        timesteps=8,
    )
    monkeypatch.setattr(train_sb3, "make_env", make_fake_env)

    episodes = train_sb3.evaluate_model(
        args, FakeModel(), training_seed=7, model_id="model-7"
    )

    assert len(episodes) == 2
    assert episodes[0].crashed is True
    assert episodes[0].notes["exception"].startswith("RuntimeError")
    assert episodes[1].success is True


def test_evaluation_cleanup_failure_is_recorded_without_dropping_episode(monkeypatch):
    from types import SimpleNamespace

    from benchmarks.external_comparison import train_sb3

    class BadCloseEnv:
        def reset(self, *, seed=None):
            return 0, {}

        def step(self, _action):
            return 0, 1.0, True, False, {"success": True, "max_pen": 0.0, "dist": 0.0}

        def close(self):
            raise RuntimeError("synthetic cleanup failure")

    class FakeModel:
        def predict(self, _obs, deterministic):
            return 0, None

    args = SimpleNamespace(
        eval_episodes=1,
        seed=0,
        environment="lumen",
        task="nav_tube",
        max_steps=1,
        algo="ppo",
        timesteps=8,
    )
    monkeypatch.setattr(train_sb3, "make_env", lambda *_args: BadCloseEnv())

    episodes = train_sb3.evaluate_model(args, FakeModel())

    assert len(episodes) == 1
    assert episodes[0].success is True
    assert episodes[0].notes["cleanup_exception"].startswith("RuntimeError")


def test_training_cleanup_failure_does_not_mask_training_error(monkeypatch):
    from types import SimpleNamespace

    from benchmarks.external_comparison import train_sb3

    class BadCloseEnv:
        def close(self):
            raise RuntimeError("synthetic cleanup failure")

    class FailingModel:
        def __init__(self, _policy, _env, **_kwargs):
            pass

        def learn(self, **_kwargs):
            raise RuntimeError("synthetic training failure")

    args = SimpleNamespace(
        environment="lumen",
        task="nav_tube",
        max_steps=1,
        algo="ppo",
        verbose=0,
        batch_size=1,
        ppo_n_steps=1,
        timesteps=1,
    )
    monkeypatch.setattr(train_sb3, "make_env", lambda *_args: BadCloseEnv())
    monkeypatch.setattr(train_sb3, "ALGOS", {"ppo": FailingModel})

    with pytest.raises(train_sb3.SeedRunError) as exc_info:
        train_sb3._train_one(args, 7, Path("model.zip"))

    assert exc_info.value.stage == "train"
    assert "synthetic training failure" in repr(exc_info.value.cause)
