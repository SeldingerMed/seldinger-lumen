from pathlib import Path


WORKFLOW = Path(".github/workflows/gpu-benchmark.yml")
BENCHMARK = Path("examples/benchmark_throughput.py")


def test_gpu_benchmark_workflow_targets_cuda_runner_and_commands():
    text = WORKFLOW.read_text()

    assert "workflow_dispatch:" in text
    assert "schedule:" in text
    assert "runs-on: [self-hosted, linux, x64, cuda]" in text
    assert "python -m lumen.hardware" in text
    assert "tests/test_newton_anatomy.py tests/test_throughput.py" in text
    assert "python examples/benchmark_throughput.py" in text
    assert "--device cuda" in text
    assert "--require-cuda" in text
    assert "--min-env-steps-per-s" in text
    assert "actions/upload-artifact@v4" in text


def test_gpu_benchmark_workflow_schedule_is_opt_in_until_runner_exists():
    text = WORKFLOW.read_text()

    assert "LUMEN_ENABLE_SCHEDULED_GPU_BENCHMARK" in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "vars.LUMEN_ENABLE_SCHEDULED_GPU_BENCHMARK == 'true'" in text


def test_throughput_script_has_ci_failure_flags():
    text = BENCHMARK.read_text()

    assert "--require-cuda" in text
    assert "--min-env-steps-per-s" in text
    assert "return 2" in text
    assert "return 0 if ok else 1" in text
