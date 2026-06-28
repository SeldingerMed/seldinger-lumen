"""Run the Layer-0 navigation benchmark and print the leaderboard (doc M5).

    python examples/run_benchmark.py [results_dir]

Evaluates the forward-advance baseline over the fixed suite, writes a scorecard to
`results_dir` (default ./bench_results), then prints the leaderboard over everything
there. Drop a `lumen.bench.evaluate_policy(your_policy, "your-name").save(...)` scorecard
in the same directory and re-run to compare — that is the "external groups can run and
submit" loop.
"""

from __future__ import annotations

import sys

from lumen.cli import benchmark_main


def main():
    benchmark_main(sys.argv[1:])


if __name__ == "__main__":
    main()
