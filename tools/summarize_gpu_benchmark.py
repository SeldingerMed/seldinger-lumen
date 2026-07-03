"""Validate and summarize CUDA hardware benchmark artifacts.

The GPU GitHub Actions job writes two machine-readable files:

* ``hardware.json`` from ``python -m lumen.hardware``
* ``gpu-throughput.json`` from ``examples/benchmark_throughput.py --json``

This helper is intentionally dependency-free so the workflow can fail closed after
benchmark execution if either artifact is malformed, reports the wrong device, or
misses the throughput threshold. It also writes a compact Markdown summary for the
GitHub Actions job summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class BenchmarkSummaryError(ValueError):
    """Raised when benchmark artifacts do not prove a passing CUDA run."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise BenchmarkSummaryError(f"missing artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkSummaryError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BenchmarkSummaryError(f"expected object JSON in {path}, got {type(data).__name__}")
    return data


def _display_value(value: Any) -> Any:
    return "N/A" if value is None else value


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkSummaryError(f"invalid {field} value: {value!r}") from exc


def _as_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkSummaryError(f"invalid {field} value: {value!r}") from exc


def _as_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkSummaryError(f"{field} must be a boolean, got {type(value).__name__}")
    return value


def _row_env_count(row: dict[str, Any], index: int) -> int:
    has_n_envs = "n_envs" in row
    has_envs = "envs" in row
    if has_n_envs == has_envs:
        raise BenchmarkSummaryError(
            f"throughput row {index} must contain exactly one of 'n_envs' or 'envs'"
        )
    field = "n_envs" if has_n_envs else "envs"
    n_envs = _as_int(row[field], f"throughput row {index} {field}")
    if n_envs <= 0:
        raise BenchmarkSummaryError(f"throughput row {index} {field}={n_envs}, expected positive")
    return n_envs


def validate_and_render_summary(hardware: dict[str, Any], throughput: dict[str, Any]) -> str:
    """Validate artifact consistency and return a Markdown benchmark summary."""

    errors: list[str] = []
    hardware_device = hardware.get("device")
    throughput_device = throughput.get("device")
    cuda_devices = _as_int(hardware.get("cuda_devices") or 0, "hardware cuda_devices")
    peak = _as_float(
        throughput.get("peak_env_steps_per_s") or 0.0,
        "throughput peak_env_steps_per_s",
    )
    target = _as_float(
        throughput.get("target_env_steps_per_s") or 0.0,
        "throughput target_env_steps_per_s",
    )
    threshold = throughput.get("min_env_steps_per_s")
    threshold_float = (
        _as_float(threshold, "throughput min_env_steps_per_s")
        if threshold is not None
        else None
    )
    passed = _as_bool(throughput.get("passed"), "throughput passed")
    newton_available = _as_bool(hardware.get("newton_available"), "hardware newton_available")
    rows = throughput.get("rows", [])
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        errors.append(f"throughput rows must be a list, got {type(rows).__name__}")

    if hardware_device != "cuda":
        errors.append(f"hardware device is {hardware_device!r}, expected 'cuda'")
    if throughput_device != "cuda":
        errors.append(f"throughput device is {throughput_device!r}, expected 'cuda'")
    if cuda_devices < 1:
        errors.append(f"cuda_devices={cuda_devices}, expected at least 1")
    if not newton_available:
        errors.append("Newton is not available in hardware report")
    if not passed:
        errors.append("throughput benchmark reported passed=false")
    if threshold_float is not None and peak < threshold_float:
        errors.append(f"peak {peak:.0f} env-steps/s is below threshold {threshold_float:.0f}")
    if not rows:
        errors.append("throughput rows are empty")

    if errors:
        raise BenchmarkSummaryError("; ".join(errors))

    lines = [
        "## Lumen CUDA hardware benchmark",
        "",
        f"- Device: `{throughput_device}` ({cuda_devices} CUDA device(s) visible)",
        f"- Warp: `{_display_value(hardware.get('warp'))}`",
        f"- Newton available: `{newton_available}`",
        f"- Peak throughput: `{peak:.0f}` env-steps/s",
        f"- Target throughput: `{target:.0f}` env-steps/s",
    ]
    if threshold_float is not None:
        lines.append(f"- Required minimum: `{threshold_float:.0f}` env-steps/s")
    lines.extend([
        "",
        "| envs | env-steps/s | ms/step | us/env-step |",
        "| ---: | ----------: | ------: | ----------: |",
    ])
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BenchmarkSummaryError(
                f"throughput row {index} must be an object, got {type(row).__name__}"
            )
        lines.append(
            "| {n_envs} | {env_steps_per_s:.0f} | {ms_per_step:.2f} | {us_per_env_step:.2f} |".format(
                n_envs=_row_env_count(row, index),
                env_steps_per_s=_as_float(
                    row.get("env_steps_per_s") or 0.0,
                    f"throughput row {index} env_steps_per_s",
                ),
                ms_per_step=_as_float(
                    row.get("ms_per_step") or 0.0,
                    f"throughput row {index} ms_per_step",
                ),
                us_per_env_step=_as_float(
                    row.get("us_per_env_step") or 0.0,
                    f"throughput row {index} us_per_env_step",
                ),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", type=Path, default=Path("hardware.json"))
    parser.add_argument("--throughput", type=Path, default=Path("gpu-throughput.json"))
    parser.add_argument("--out", type=Path, default=Path("gpu-benchmark-summary.md"))
    args = parser.parse_args(argv)

    hardware = _load_json(args.hardware)
    throughput = _load_json(args.throughput)
    summary = validate_and_render_summary(hardware, throughput)
    try:
        args.out.write_text(summary)
    except OSError as exc:
        raise BenchmarkSummaryError(f"failed to write summary: {exc}") from exc
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
