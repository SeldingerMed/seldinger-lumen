"""Regression checks for native endpoint reporting follow-ups."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_lumen_reset_exception_is_a_failed_native_pass(tmp_path, monkeypatch):
    common = _load_module(
        "common_bench_followup_test",
        ROOT / "benchmarks" / "external_comparison" / "common_bench.py",
    )

    class ResetFailureEnv:
        action_space = None

        def reset(self, *, seed=None):
            raise RuntimeError("synthetic reset failure")

    from lumen.envs import registration

    monkeypatch.setattr(registration, "make_nav_tube", lambda **_: ResetFailureEnv())
    args = argparse.Namespace(
        max_steps=1,
        policies="forward",
        tasks="nav_tube",
        episodes=1,
        seed=0,
        progress=False,
        out_dir=str(tmp_path),
        run_id="reset_failure",
    )
    common.run_lumen(args)
    payload = json.loads((tmp_path / "reset_failure.json").read_text())
    episode = payload["episodes"][0]
    assert episode["native_safety_pass"] is False
    assert episode["crashed"] is True
    assert payload["aggregate"][0]["native_safety_pass_rate"] == 0.0


def test_external_contract_and_csv_keep_emitted_return_and_wall_fields():
    contract = json.loads(
        (ROOT / "benchmarks" / "external_comparison" / "metric_contract.json").read_text()
    )
    assert "mean_return" in contract["required_metrics"]["higher_is_better"]
    assert "mean_return_loss" not in json.dumps(contract)

    summary = _load_module(
        "summary_followup_test",
        ROOT / "benchmarks" / "external_comparison" / "summarize_results.py",
    )
    assert "max_wall_pressure" in summary.KEYS
    assert "wall_load_impulse" in summary.KEYS


def test_comparison_bullet_is_complete():
    text = (ROOT / "docs" / "comparison.md").read_text()
    assert "interaction, and fragmentation." in text
