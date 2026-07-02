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


def validate_and_render_summary(hardware: dict[str, Any], throughput: dict[str, Any]) -> str:
    """Validate artifact consistency and return a Markdown benchmark summary."""

    errors: list[str] = []
    hardware_device = hardware.get("device")
    throughput_device = throughput.get("device")
    cuda_devices = int(hardware.get("cuda_devices") or 0)
    peak = float(throughput.get("peak_env_steps_per_s") or 0.0)
    target = float(throughput.get("target_env_steps_per_s") or 0.0)
    threshold = throughput.get("min_env_steps_per_s")
    threshold_float = float(threshold) if threshold is not None else None
    passed = bool(throughput.get("passed"))
    rows = throughput.get("rows") or []

    if hardware_device != "cuda":
        errors.append(f"hardware device is {hardware_device!r}, expected 'cuda'")
    if throughput_device != "cuda":
        errors.append(f"throughput device is {throughput_device!r}, expected 'cuda'")
    if cuda_devices < 1:
        errors.append(f"cuda_devices={cuda_devices}, expected at least 1")
    if not hardware.get("newton_available"):
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
        f"- Warp: `{hardware.get('warp')}`",
        f"- Newton available: `{hardware.get('newton_available')}`",
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
    for row in rows:
        lines.append(
            "| {n_envs} | {env_steps_per_s:.0f} | {ms_per_step:.2f} | {us_per_env_step:.2f} |".format(
                n_envs=int(row.get("n_envs") or row.get("envs") or 0),
                env_steps_per_s=float(row.get("env_steps_per_s") or 0.0),
                ms_per_step=float(row.get("ms_per_step") or 0.0),
                us_per_env_step=float(row.get("us_per_env_step") or 0.0),
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
    args.out.write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
