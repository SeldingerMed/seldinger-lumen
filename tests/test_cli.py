"""Installed command entry points for first-run workflows."""

from importlib.metadata import entry_points


def test_pyproject_exposes_first_run_console_scripts():
    scripts = {
        ep.name: ep.value
        for ep in entry_points(group="console_scripts")
        if ep.name.startswith("lumen-")
    }

    assert scripts == {
        "lumen-hardware": "lumen.cli:hardware_main",
        "lumen-benchmark": "lumen.cli:benchmark_main",
        "lumen-replay": "lumen.cli:replay_main",
        "lumen-calibrate": "lumen.cli:calibrate_main",
    }


def test_replay_cli_handles_missing_root_without_warning(tmp_path, capsys):
    import warnings

    from lumen.cli import replay_main

    missing = tmp_path / "missing"
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        replay_main([str(missing)])

    out = capsys.readouterr().out
    assert "run examples/capture_episode.py first" in out
    assert seen == []
