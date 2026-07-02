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
    ],
)
def test_validate_and_render_summary_fails_closed(hardware, throughput, message):
    with pytest.raises(BenchmarkSummaryError, match=message):
        validate_and_render_summary(hardware, throughput)


def test_main_writes_markdown_summary(tmp_path: Path):
    hardware = tmp_path / "hardware.json"
    throughput = tmp_path / "gpu-throughput.json"
    out = tmp_path / "summary.md"
    hardware.write_text(json.dumps(_hardware()))
    throughput.write_text(json.dumps(_throughput()))

    assert main(["--hardware", str(hardware), "--throughput", str(throughput), "--out", str(out)]) == 0
    assert out.exists()
    assert "Required minimum: `10000` env-steps/s" in out.read_text()
