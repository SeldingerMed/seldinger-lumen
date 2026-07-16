"""Summarize common endovascular benchmark result files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


KEYS = [
    "environment",
    "task",
    "policy",
    "episodes",
    "success_rate",
    "safe_success_rate",
    "crash_rate",
    "unsafe_event_rate",
    "mean_steps_success",
    "mean_steps_all",
    "mean_final_distance",
    "mean_return",
    "max_contact_force",
    "mean_contact_force",
    "steps_per_second",
]


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("aggregate", []):
            row = dict(row)
            row["source_file"] = str(path)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.results)
    print(json.dumps(rows, indent=2))
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=KEYS + ["source_file"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
