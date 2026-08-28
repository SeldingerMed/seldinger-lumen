"""Evaluate and replay-verify the canonical benchmark baseline."""

from __future__ import annotations

import json

from lumen.bench import evaluate_policy, forward_policy, verify_replay


def main() -> None:
    card = evaluate_policy(forward_policy, "forward-baseline")
    report = verify_replay(card, forward_policy)
    if not report["verified"]:
        raise RuntimeError(f"benchmark replay verification failed: {report['errors']}")
    print(json.dumps({
        "policy": card.name,
        "suite_version": card.suite_version,
        "replay": report,
    }, indent=2))


if __name__ == "__main__":
    main()
