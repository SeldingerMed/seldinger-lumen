"""Run the Layer-0 navigation benchmark and print the leaderboard (doc M5).

    python examples/run_benchmark.py [results_dir]

Evaluates the forward-advance baseline over the fixed suite, writes a scorecard to
`results_dir` (default ./bench_results), then prints the leaderboard over everything
there. Drop a `lumen.bench.evaluate_policy(your_policy, "your-name").save(...)` scorecard
in the same directory and re-run to compare — that is the "external groups can run and
submit" loop.
"""

from __future__ import annotations

import os
import sys

from lumen.bench import evaluate_policy, forward_policy, leaderboard, scorecard_rejections


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "bench_results"
    os.makedirs(results_dir, exist_ok=True)

    sc = evaluate_policy(forward_policy, "forward-baseline")
    sc.save(os.path.join(results_dir, "forward-baseline.json"))

    print(f"suite {sc.suite_version}   submission: {sc.name}")
    print(f"{'task':18} {'tier':7} {'safe':>8} {'success':>8} {'mean_steps':>11} {'max_pen':>9}")
    for t in sc.per_task:
        steps = "-" if t["mean_steps"] is None else f"{t['mean_steps']:.1f}"
        print(f"{t['name']:18} {t['tier']:7} {t['safe_success_rate']:>8.2f} "
              f"{t['success_rate']:>8.2f} {steps:>11} {t['max_pen']:>9.3f}")
    o = sc.overall
    print(f"\noverall: safe={o['safe_success_rate']:.2f}  success={o['success_rate']:.2f}  "
          f"worst max_pen={o['max_pen']:.3f}  "
          f"mean_return={o['mean_return']:.1f}")

    print(f"\nleaderboard ({results_dir}):")
    for rank, c in enumerate(leaderboard(results_dir), 1):
        print(f"  {rank}. {c.name:24} safe={c.overall.get('safe_success_rate', 0.0):.2f}  "
              f"success={c.overall['success_rate']:.2f}  "
              f"max_pen={c.overall['max_pen']:.3f}")
    skipped = scorecard_rejections(results_dir)
    if skipped:
        print("\nskipped scorecards:")
        for item in skipped:
            print(f"  {item['path']}: {item['error']}")


if __name__ == "__main__":
    main()
