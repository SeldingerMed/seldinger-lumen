import json
from pathlib import Path

import pytest

from tools.summarize_gpu_benchmark import BenchmarkSummaryError, main, validate_and_render_summary


def _hardware(**overrides):
    data = {
        "device": "cuda",
        "warp": "1.14.0",
        "cuda_devices": 1,
        "newton_available": True,
    }
    data.update(overrides)
    return data


def _throughput(**overrides):
    data = {
        "device": "cuda",
        "peak_env_steps_per_s": 12500.0,
        "target_env_steps_per_s": 10000.0,
        "min_env_steps_per_s": 10000.0,
        "passed": True,
        "rows": [
            {
                "n_envs": 256,
                "env_steps_per_s": 9000.0,
                "ms_per_step": 28.4,
                "us_per_env_step": 111.1,
            },
            {
                "n_envs": 1024,
                "env_steps_per_s": 12500.0,
                "ms_per_step": 81.9,
                "us_per_env_step": 80.0,
            },
        ],
    }
    data.update(overrides)
    return data


def test_validate_and_render_summary_requires_cuda_and_threshold():
    summary = validate_and_render_summary(_hardware(), _throughput())

    assert "Lumen CUDA hardware benchmark" in summary
    assert "Peak throughput: `12500` env-steps/s" in summary
    assert "| 1024 | 12500 | 81.90 | 80.00 |" in summary


@pytest.mark.parametrize(
    ("hardware", "throughput", "message"),
    [
        (_hardware(device="cpu"), _throughput(), "hardware device"),
        (_hardware(cuda_devices=0), _throughput(), "cuda_devices=0"),
        (_hardware(newton_available=False), _throughput(), "Newton is not available"),
        (_hardware(), _throughput(device="cpu"), "throughput device"),
        (_hardware(), _throughput(passed=False), "passed=false"),
        (_hardware(), _throughput(peak_env_steps_per_s=9999.0), "below threshold"),
        (_hardware(), _throughput(rows=[]), "rows are empty"),
        (_hardware(), _throughput(rows="not-a-list"), "rows must be a list"),
    ],
)
def test_validate_and_render_summary_fails_closed(hardware, throughput, message):
    with pytest.raises(BenchmarkSummaryError, match=message):
        validate_and_render_summary(hardware, throughput)


@pytest.mark.parametrize(
    ("hardware", "throughput", "message"),
    [
        (_hardware(cuda_devices="many"), _throughput(), "hardware cuda_devices"),
        (_hardware(), _throughput(peak_env_steps_per_s="fast"), "peak_env_steps_per_s"),
        (_hardware(), _throughput(target_env_steps_per_s="fast"), "target_env_steps_per_s"),
        (_hardware(), _throughput(min_env_steps_per_s="fast"), "min_env_steps_per_s"),
        (_hardware(), _throughput(rows=[{"n_envs": "many"}]), "row 0 n_envs"),
        (_hardware(), _throughput(rows=[{"n_envs": 1, "env_steps_per_s": "fast"}]), "env_steps_per_s"),
        (_hardware(), _throughput(rows=[{"n_envs": 1, "ms_per_step": "slow"}]), "ms_per_step"),
        (_hardware(), _throughput(rows=[{"n_envs": 1, "us_per_env_step": "slow"}]), "us_per_env_step"),
    ],
)
def test_validate_and_render_summary_reports_invalid_numeric_values(hardware, throughput, message):
    with pytest.raises(BenchmarkSummaryError, match=message):
        validate_and_render_summary(hardware, throughput)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"envs": 256, "env_steps_per_s": 9000.0}, "| 256 | 9000 | 0.00 | 0.00 |"),
        ({"n_envs": 0}, "n_envs=0"),
        ({"n_envs": 256, "envs": 256}, "exactly one"),
        ({"env_steps_per_s": 9000.0}, "exactly one"),
        ("not-a-row", "row 0 must be an object"),
    ],
)
def test_validate_and_render_summary_validates_throughput_rows(row, message):
    throughput = _throughput(rows=[row])

    if isinstance(row, dict) and set(row) == {"envs", "env_steps_per_s"}:
        assert message in validate_and_render_summary(_hardware(), throughput)
    else:
        with pytest.raises(BenchmarkSummaryError, match=message):
            validate_and_render_summary(_hardware(), throughput)


def test_validate_and_render_summary_uses_placeholder_for_missing_optional_hardware():
    summary = validate_and_render_summary(_hardware(warp=None), _throughput())

    assert "Warp: `N/A`" in summary


def test_main_writes_markdown_summary(tmp_path: Path):
    hardware = tmp_path / "hardware.json"
    throughput = tmp_path / "gpu-throughput.json"
    out = tmp_path / "summary.md"
    hardware.write_text(json.dumps(_hardware()))
    throughput.write_text(json.dumps(_throughput()))

    assert main(["--hardware", str(hardware), "--throughput", str(throughput), "--out", str(out)]) == 0
    assert out.exists()
    assert "Required minimum: `10000` env-steps/s" in out.read_text()
