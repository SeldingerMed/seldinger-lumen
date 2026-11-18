#!/usr/bin/env bash
# Single-command blind-first navigation report + replay certificate.
#
# From a clean checkout of this repository:
#
#     bash tools/image_navigation.sh [--out DIR]
#
# What it does:
#   1. `uv sync --locked --extra solver` — installs the exact uv.lock
#      resolution, which pins Newton (newton-physics @ 6dfe7303) and
#      warp-lang 1.16.0 on a Python >= 3.10 interpreter.
#   2. Runs `lumen.experiments.navigation_report` on the CPU only
#      (CUDA_VISIBLE_DEVICES is cleared; the rollout engine is already
#      device="cpu") for every frozen case.
#   3. Writes report.html + certificate.json into a fresh, never-clobbered
#      directory and prints the path; then re-verifies the certificate by
#      replaying all three cases.
#
# Verify an existing certificate without regenerating:
#
#     bash tools/image_navigation.sh --verify /path/to/certificate.json
set -euo pipefail

cd "$(dirname "$0")/.."

# Force CPU execution end-to-end (Newton/Warp must never touch a GPU here).
export CUDA_VISIBLE_DEVICES=""
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"

uv sync --locked --python 3.12 --extra solver

exec uv run --no-sync python -m lumen.experiments.navigation_report "$@"
